import os
import uvicorn
import sqlite3
import logging
import threading
import json
import requests

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langdetect import detect

# ---------------- CONFIGURACIÓN ----------------
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BOT_ACTIVE = os.getenv("BOT_ACTIVE", "true")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI(title="AI Hotel Booking Agent")

INDEX_FILE = "hotel_faiss_index"
DOCUMENTS_DIR = os.path.join(os.getcwd(), "documents")
DATABASE = "memory.db"

vector_db = None
index_lock = threading.Lock()

class Message(BaseModel):
    user_id: str
    message: str

# ---------------- BASE DE DATOS ----------------
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        role TEXT,
        message TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS lead_scores (
        user_id TEXT PRIMARY KEY,
        score INTEGER DEFAULT 0
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customer_profile (
        user_id TEXT PRIMARY KEY,
        language TEXT,
        last_intent TEXT,
        last_emotion TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_message(user_id, role, message):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO conversations (user_id, role, message) VALUES (?, ?, ?)", (user_id, role, message))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Error guardando mensaje: {e}")

def get_history(user_id, limit=8):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, message FROM conversations WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    rows.reverse()
    
    history = []
    for role, message in rows:
        if role == "user":
            history.append(HumanMessage(content=message))
        else:
            history.append(AIMessage(content=message))
    return history

# ---------------- ANALÍTICA ----------------
def analyze_customer_behavior(message):
    llm_analyst = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""
    Analiza el siguiente mensaje y responde ÚNICAMENTE con el JSON.
    Mensaje: "{message}"
    JSON: {{"intencion":"valor","emocion":"valor"}}
    """
    try:
        response = llm_analyst.invoke(prompt)
        clean_content = response.content.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean_content)
        return data.get("intencion", "informacion"), data.get("emocion", "neutral")
    except Exception as e:
        logging.warning(f"Fallo análisis semántico: {e}")
        return "informacion", "neutral"

def update_lead_data(user_id, points, language, intent, emotion):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO lead_scores (user_id, score) VALUES (?, 0)", (user_id,))
    cursor.execute("UPDATE lead_scores SET score = score + ? WHERE user_id = ?", (points, user_id))
    cursor.execute("""
    INSERT OR REPLACE INTO customer_profile (user_id, language, last_intent, last_emotion)
    VALUES (?, ?, ?, ?)
    """, (user_id, language, intent, emotion))
    conn.commit()
    cursor.execute("SELECT score FROM lead_scores WHERE user_id=?", (user_id,))
    score = cursor.fetchone()[0]
    conn.close()
    return score

# ---------------- MOTOR IA (CORREGIDO) ----------------
def process_message(user_id, message):
    try:
        if str(BOT_ACTIVE).lower() != "true":
            return "El asistente está temporalmente desactivado."

        intent, emotion = analyze_customer_behavior(message)
        language = detect(message) if len(message) > 3 else "es"
        
        # Puntos de interés comercial
        points = 25 if intent in ["reserva", "precio", "disponibilidad"] else 5
        total_score = update_lead_data(user_id, points, language, intent, emotion)

        save_message(user_id, "user", message)
        history = get_history(user_id)

        # Búsqueda de información en PDFs
        contexto = ""
        if vector_db is not None:
            with index_lock:
                docs = vector_db.similarity_search(message, k=4)
                contexto = "\n\n".join([d.page_content for d in docs])

        if not contexto:
            contexto = "No hay detalles específicos en los documentos, pero intenta ser servicial y pide datos para contactar al cliente."

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
        
        # MEJORA DEL SYSTEM PROMPT: Personalidad de Vendedor
        system_rules = f"""
        Eres el Asistente Virtual de Reservas oficial. Tu única misión es informar y VENDER habitaciones o servicios basándote en la información de los documentos.

        DIRECTRICES DE COMPORTAMIENTO:
        1. Identidad: Eres parte del equipo del hotel/negocio, no un consultor externo de ventas. 
        2. Proactividad: Si el cliente saluda, dale la bienvenida con entusiasmo y menciona algo atractivo que esté en el contexto.
        3. Uso de Datos: Extrae precios, tipos de habitación y amenidades directamente del CONTEXTO proporcionado abajo.
        4. Cierre de Venta: Si el score del cliente ({total_score}) es mayor a 60, solicita amablemente sus fechas de viaje y correo para formalizar.
        5. Idioma: Responde siempre en {language}.

        REGLA DE ORO: No preguntes "en qué aspecto de ventas necesitas ayuda". Pregunta "¿Cuándo te gustaría hospedarte con nosotros?" o "¿Qué tipo de habitación buscas?".

        CONTEXTO DISPONIBLE:
        {contexto}
        """

        messages = [
            SystemMessage(content=system_rules),
            *history,
            HumanMessage(content=message)
        ]

        response = llm.invoke(messages)
        save_message(user_id, "assistant", response.content)
        return response.content

    except Exception as e:
        logging.error(f"Error en process_message: {e}")
        return "Lo siento, tuve un inconveniente técnico. ¿Me puedes repetir tu pregunta?"

# ---------------- RAG E INDEXACIÓN ----------------
def build_index():
    global vector_db
    try:
        embeddings = OpenAIEmbeddings()
        if os.path.exists(INDEX_FILE):
            vector_db = FAISS.load_local(INDEX_FILE, embeddings, allow_dangerous_deserialization=True)
            logging.info("FAISS cargado.")
        else:
            if not os.path.exists(DOCUMENTS_DIR): os.makedirs(DOCUMENTS_DIR)
            docs = []
            for file in os.listdir(DOCUMENTS_DIR):
                if file.endswith(".pdf"):
                    logging.info(f"Procesando PDF: {file}")
                    loader = PyPDFLoader(os.path.join(DOCUMENTS_DIR, file))
                    docs.extend(loader.load())
            
            if docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                vector_db = FAISS.from_documents(chunks, embeddings)
                vector_db.save_local(INDEX_FILE)
                logging.info("Nuevo índice FAISS creado exitosamente.")
            else:
                logging.warning("No hay PDFs en la carpeta /documents. Sube archivos para que la IA tenga información.")
    except Exception as e:
        logging.error(f"Error en build_index: {e}")

# ---------------- ENDPOINTS ----------------
@app.on_event("startup")
async def startup():
    init_db()
    threading.Thread(target=build_index, daemon=True).start()

@app.post("/telegram")
async def telegram_webhook(request: Request):
    data = await request.json()
    try:
        chat_msg = data.get("message", {})
        text = chat_msg.get("text")
        chat_id = chat_msg.get("chat", {}).get("id")

        if text and chat_id:
            reply = process_message(str(chat_id), text)
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": reply})
    except Exception as e:
        logging.error(f"Error Webhook: {e}")
    return {"ok": True}

@app.get("/")
def root(): return {"status": "AI Booking Agent is Online"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
