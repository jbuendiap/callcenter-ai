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

# TOKEN Y VARIABLES DE RAILWAY
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
# Usamos un valor por defecto seguro para evitar errores de NoneType
BOT_ACTIVE = os.getenv("BOT_ACTIVE", "true")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI(title="AI Call Center Pro")

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

# ---------------- ANALÍTICA MEJORADA ----------------
def analyze_customer_behavior(message):
    llm_analyst = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Prompt más estricto para evitar texto basura fuera del JSON
    prompt = f"""
    Analiza el siguiente mensaje y responde ÚNICAMENTE con el JSON.
    Mensaje: "{message}"
    JSON: {{"intencion":"valor","emocion":"valor"}}
    """
    try:
        response = llm_analyst.invoke(prompt)
        # Limpieza de Markdown si la IA lo incluye
        clean_content = response.content.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean_content)
        return data.get("intencion", "informacion"), data.get("emocion", "neutral")
    except Exception as e:
        logging.warning(f"Fallo análisis semántico: {e}")
        return "informacion", "neutral"

def update_lead_data(user_id, points, language, intent, emotion):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    # Usar INSERT OR IGNORE para asegurar que el registro exista antes del UPDATE
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

# ---------------- MOTOR IA ----------------
def process_message(user_id, message):
    try:
        # Validación segura de variable de entorno
        if str(BOT_ACTIVE).lower() != "true":
            return "El asistente está temporalmente desactivado."

        intent, emotion = analyze_customer_behavior(message)
        language = detect(message) if len(message) > 3 else "es"
        
        # Puntos básicos
        points = 20 if "reservar" in message.lower() else 5
        total_score = update_lead_data(user_id, points, language, intent, emotion)

        save_message(user_id, "user", message)
        history = get_history(user_id)

        contexto = "No hay información adicional."
        if vector_db is not None:
            with index_lock:
                docs = vector_db.similarity_search(message, k=4)
                contexto = "\n\n".join([d.page_content for d in docs])

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
        
        messages = [
            SystemMessage(content=f"Eres experto en ventas. Idioma: {language}. Contexto: {contexto}"),
            *history,
            HumanMessage(content=message)
        ]

        response = llm.invoke(messages)
        save_message(user_id, "assistant", response.content)
        return response.content

    except Exception as e:
        logging.error(f"Error en process_message: {e}")
        return "Lo siento, tuve un problema técnico. ¿Podrías repetir eso?"

# ---------------- RAG E INDEXACIÓN ----------------
def build_index():
    global vector_db
    try:
        embeddings = OpenAIEmbeddings()
        if os.path.exists(INDEX_FILE):
            vector_db = FAISS.load_local(INDEX_FILE, embeddings, allow_dangerous_deserialization=True)
            logging.info("FAISS cargado desde disco.")
        else:
            if not os.path.exists(DOCUMENTS_DIR): os.makedirs(DOCUMENTS_DIR)
            # Aquí podrías cargar PDFs...
            logging.warning("No se encontró índice FAISS. Por favor sube PDFs a /documents.")
    except Exception as e:
        logging.error(f"Error creando índice: {e}")

# ---------------- ENDPOINTS ----------------
@app.on_event("startup")
async def startup():
    init_db()
    threading.Thread(target=build_index, daemon=True).start()

@app.post("/telegram")
async def telegram_webhook(request: Request):
    data = await request.json()
    try:
        # Extraer datos de Telegram
        chat_msg = data.get("message", {})
        text = chat_msg.get("text")
        chat_id = chat_msg.get("from", {}).get("id")

        if not text or not chat_id:
            return {"ok": True}

        # Procesar
        reply = process_message(str(chat_id), text)

        # Enviar respuesta
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": reply})
        
    except Exception as e:
        logging.error(f"Error Webhook Telegram: {e}")
    
    return {"ok": True}

@app.get("/")
def root(): return {"status": "Online"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
