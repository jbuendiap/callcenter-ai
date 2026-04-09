import os
import uvicorn
import sqlite3
import logging
import threading
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langdetect import detect

# ---------------- CONFIGURACIÓN ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI(title="AI Call Center Pro - Lead Scoring & Emotion AI")

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
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversations (user_id, role, message) VALUES (?, ?, ?)", (user_id, role, message))
    conn.commit()
    conn.close()

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
            history.append(SystemMessage(content=message))
    return history

# ---------------- ANALÍTICA CON IA (EMOCIÓN E INTENCIÓN) ----------------
def analyze_customer_behavior(message):
    """
    Usa gpt-4o-mini para clasificar semánticamente al cliente.
    """
    llm_analyst = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""
    Analiza el siguiente mensaje de un cliente y clasifícalo en dos categorías.
    Mensaje: "{message}"

    Responde ÚNICAMENTE en este formato JSON:
    {{"intencion": "valor", "emocion": "valor"}}

    Opciones de intención: [saludo, precio, disponibilidad, reserva, objecion, comparacion, despedida, informacion]
    Opciones de emoción: [frustracion, satisfaccion, decision_compra, duda, neutral]
    """
    try:
        response = llm_analyst.invoke(prompt)
        data = json.loads(response.content.strip())
        return data.get("intencion", "informacion"), data.get("emocion", "neutral")
    except Exception as e:
        logging.warning(f"Error al analizar comportamiento: {e}")
        return "informacion", "neutral"

def update_lead_data(user_id, points, language, intent, emotion):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    # Actualizar Score
    cursor.execute("""
    INSERT INTO lead_scores (user_id, score) VALUES(?, ?)
    ON CONFLICT(user_id) DO UPDATE SET score = score + excluded.score
    """, (user_id, points))
    # Actualizar Perfil
    cursor.execute("""
    INSERT OR REPLACE INTO customer_profile (user_id, language, last_intent, last_emotion)
    VALUES (?, ?, ?, ?)
    """, (user_id, language, intent, emotion))
    conn.commit()
    
    cursor.execute("SELECT score FROM lead_scores WHERE user_id=?", (user_id,))
    final_score = cursor.fetchone()[0]
    conn.close()
    return final_score

def calculate_points(intent, emotion):
    # Puntos más granular
    points = 5 # Puntos por interactuar
    intent_points = {"precio":15, "reserva":40, "disponibilidad":20, "objecion":-10, "comparacion":5, "saludo":2}
    emotion_points = {"decision_compra":30, "frustracion":-10, "satisfaccion":15, "duda":-5}
    points += intent_points.get(intent,0)
    points += emotion_points.get(emotion,0)
    return points

# ---------------- MOTOR RAG (FAISS) ----------------
def build_index():
    global vector_db
    try:
        embeddings = OpenAIEmbeddings()
        if os.path.exists(INDEX_FILE):
            vector_db = FAISS.load_local(INDEX_FILE, embeddings, allow_dangerous_deserialization=True)
            logging.info("Índice FAISS cargado.")
        else:
            if not os.path.exists(DOCUMENTS_DIR): os.makedirs(DOCUMENTS_DIR)
            docs = []
            for file in os.listdir(DOCUMENTS_DIR):
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(DOCUMENTS_DIR, file))
                    docs.extend(loader.load())
            if docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                vector_db = FAISS.from_documents(chunks, embeddings)
                vector_db.save_local(INDEX_FILE)
                logging.info("Índice FAISS creado.")
    except Exception as e:
        logging.error(f"Error FAISS: {e}")

# ---------------- ENDPOINT PRINCIPAL ----------------
@app.on_event("startup")
async def startup():
    init_db()
    threading.Thread(target=build_index, daemon=True).start()

@app.post("/chat")
async def chat(data: Message):
    if vector_db is None:
        return {"response": "Iniciando sistema... intente en 5 segundos."}

    try:
        # 1. Análisis Semántico y Lead Scoring
        intent, emotion = analyze_customer_behavior(data.message)
        language = detect(data.message) if len(data.message) > 3 else "es"
        points = calculate_points(intent, emotion)
        total_score = update_lead_data(data.user_id, points, language, intent, emotion)

        # 2. Contexto RAG y Memoria
        save_message(data.user_id, "user", data.message)
        history = get_history(data.user_id)
        
        with index_lock:
            docs = vector_db.similarity_search(data.message, k=4)
        contexto = "\n\n".join([doc.page_content for doc in docs])

        # 3. Prompt Dinámico según comportamiento
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
        
        system_rules = f"""
        Eres un experto en ventas. Responde solo con el contexto proporcionado.
        Idioma: {language} | Intención detectada: {intent} | Emoción: {emotion}
        Score del cliente: {total_score}

        REGLA DE ORO: Si no sabes la respuesta, di: "Esta información la consultaré y le responderé en la brevedad."
        """

        if total_score >= 70 or intent == "reserva" or emotion == "decision_compra":
            system_rules += "\nEL CLIENTE ESTÁ LISTO: Solicita Nombre, Fechas, Personas y Contacto de forma directa."

        messages = [
            SystemMessage(content=f"{system_rules}\n\nCONTEXTO:\n{contexto}"),
            *history,
            HumanMessage(content=data.message)
        ]

        # 4. Respuesta y Guardado
        response = llm.invoke(messages)
        save_message(data.user_id, "assistant", response.content)

        return {
            "response": response.content,
            "analytics": {
                "score": total_score,
                "intent": intent,
                "emotion": emotion,
                "language": language
            }
        }
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
