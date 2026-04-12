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

from langdetect import detect, DetectorFactory

# Garantiza que los resultados de detección de idioma sean consistentes
DetectorFactory.seed = 0

# ---------------- CONFIGURACIÓN ----------------
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BOT_ACTIVE = os.getenv("BOT_ACTIVE", "true")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI(title="AI Hotel Elite Sales Agent - Multilang")

INDEX_FILE = "hotel_faiss_index"
DOCUMENTS_DIR = os.path.join(os.getcwd(), "documents")
DATABASE = "memory.db"

vector_db = None
index_lock = threading.Lock()

# ---------------- BASE DE DATOS ----------------
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT, role TEXT, message TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("CREATE TABLE IF NOT EXISTS lead_scores (user_id TEXT PRIMARY KEY, score INTEGER DEFAULT 0)")
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
        logging.error(f"Error DB: {e}")

def get_history(user_id, limit=6):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, message FROM conversations WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    rows.reverse()
    return [HumanMessage(content=m) if r == "user" else AIMessage(content=m) for r, m in rows]

# ---------------- MOTOR IA (VENDEDOR MULTILENGUAJE) ----------------
def process_message(user_id, message):
    try:
        if str(BOT_ACTIVE).lower() != "true":
            return "System offline."

        # DETECCIÓN DE IDIOMA MEJORADA
        try:
            lang_code = detect(message)
        except:
            lang_code = "es"

        # Mapeo extendido para incluir RUSO y otros
        lang_map = {
            "es": "Español", "en": "Inglés", "ru": "Ruso", 
            "fr": "Francés", "de": "Alemán", "it": "Italiano"
        }
        idioma_destino = lang_map.get(lang_code, "Español")

        save_message(user_id, "user", message)
        history = get_history(user_id)

        contexto = ""
        if vector_db is not None:
            with index_lock:
                # k=10 para capturar bien las objeciones
                docs = vector_db.similarity_search(message, k=10)
                contexto = "\n\n".join([d.page_content for d in docs])

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        
        system_rules = f"""
        ROL: Eres un Experto Vendedor de Lujo. 
        IDIOMA DE RESPUESTA: Debes responder OBLIGATORIAMENTE en {idioma_destino}.
        
        TU MISIÓN:
        1. Vender la opción más cara y mejorar la experiencia del cliente (Upselling).
        2. Usar el CONTEXTO proporcionado para manejar objeciones. Los documentos están en Español, pero tú debes traducir las soluciones al {idioma_destino} de forma fluida y persuasiva.
        3. Si el cliente habla en Ruso, responde con gramática perfecta en Ruso.
        4. Sé un cerrador de ventas, no un informador.

        CONTEXTO:
        {contexto}
        """

        messages = [SystemMessage(content=system_rules), *history, HumanMessage(content=message)]
        response = llm.invoke(messages)
        save_message(user_id, "assistant", response.content)
        return response.content

    except Exception as e:
        logging.error(f"Error: {e}")
        return "Internal Error."

# ---------------- RAG E INDEXACIÓN ----------------
def build_index():
    global vector_db
    try:
        embeddings = OpenAIEmbeddings()
        if not os.path.exists(DOCUMENTS_DIR): os.makedirs(DOCUMENTS_DIR)
        
        docs = []
        for file in os.listdir(DOCUMENTS_DIR):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(DOCUMENTS_DIR, file))
                docs.extend(loader.load())
        
        if docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            with index_lock:
                vector_db = FAISS.from_documents(chunks, embeddings)
                vector_db.save_local(INDEX_FILE)
            logging.info("FAISS Index Rebuilt.")
    except Exception as e:
        logging.error(f"Index Error: {e}")

# ---------------- ENDPOINTS ----------------

@app.post("/vapi-webhook")
async def vapi_webhook(request: Request):
    data = await request.json()
    try:
        tool_calls = data.get("message", {}).get("toolCalls", [])
        if tool_calls:
            tool_call = tool_calls[0]
            args = tool_call.get("function", {}).get("arguments", {})
            user_query = args.get("query") or args.get("message") or ""
            if user_query:
                respuesta = process_message("vapi_user", user_query)
                return {"results": [{"toolCallId": tool_call.get("id"), "result": respuesta}]}
    except: pass
    return {"ok": True}

@app.on_event("startup")
async def startup():
    init_db()
    threading.Thread(target=build_index, daemon=True).start()

@app.get("/")
def root(): return {"status": "Online"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
