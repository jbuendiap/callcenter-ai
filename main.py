import os
import uvicorn
import sqlite3
import logging
import threading
import json
import requests
import shutil

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langdetect import detect, DetectorFactory

# Estabilidad para idiomas como el Ruso
DetectorFactory.seed = 0

# ---------------- CONFIGURACIÓN ----------------
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BOT_ACTIVE = os.getenv("BOT_ACTIVE", "true")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI(title="AI Hotel Elite Sales Agent - Full Correction")

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

# ---------------- MOTOR IA (VENDEDOR ELITE MULTILENGUAJE) ----------------
def process_message(user_id, message):
    try:
        if str(BOT_ACTIVE).lower() != "true":
            return "Lo sentimos, el sistema de reservas está en mantenimiento."

        # DETECCIÓN DE IDIOMA
        try:
            lang_code = detect(message)
        except:
            lang_code = "es"

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
                # k=12 para encontrar precios y quejas sin fallar
                docs = vector_db.similarity_search(message, k=12)
                contexto = "\n\n".join([f"FUENTE: {d.page_content}" for d in docs])

        if not contexto:
            contexto = "No se encontró información en los documentos. Pide al cliente esperar un momento."

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        
        system_rules = f"""
        ROL: Eres el Gerente de Ventas Senior. Eres persuasivo, elegante y directo.
        IDIOMA: Debes responder OBLIGATORIAMENTE en {idioma_destino}.
        
        INSTRUCCIONES DE BÚSQUEDA CRÍTICAS:
        1. PRECIOS: Busca en el CONTEXTO cualquier cifra con '$', 'USD' o 'costo'. Si el cliente pregunta precio, dale el valor exacto que aparece en 'hotel_info.pdf'.
        2. QUEJAS: Si el cliente está molesto o duda, usa las técnicas de 'manejo_objeciones.pdf' que están abajo.
        3. FUENTE DE VERDAD: Tu conocimiento viene SOLO de los manuales. Si la info está en el CONTEXTO, no puedes decir "no sé".
        4. UPSELLING: Siempre sugiere una mejora de habitación o servicio VIP basándote en los documentos.

        CONTEXTO RECUPERADO DE LOS MANUALES:
        {contexto}
        """

        messages = [SystemMessage(content=system_rules), *history, HumanMessage(content=message)]
        response = llm.invoke(messages)
        save_message(user_id, "assistant", response.content)
        return response.content

    except Exception as e:
        logging.error(f"Error en el proceso: {e}")
        return "Disculpe, estoy teniendo dificultades técnicas. ¿Podría repetir su pregunta?"

# ---------------- RAG E INDEXACIÓN (MEJORADA) ----------------
def build_index():
    global vector_db
    try:
        embeddings = OpenAIEmbeddings()
        
        # ELIMINAR ÍNDICE VIEJO PARA FORZAR ACTUALIZACIÓN DE PRECIOS
        if os.path.exists(INDEX_FILE):
            logging.info("Borrando índice antiguo para actualizar datos...")
            shutil.rmtree(INDEX_FILE)

        if not os.path.exists(DOCUMENTS_DIR): 
            os.makedirs(DOCUMENTS_DIR)
        
        docs = []
        for file in os.listdir(DOCUMENTS_DIR):
            if file.endswith(".pdf"):
                logging.info(f"Cargando documento: {file}")
                loader = PyPDFLoader(os.path.join(DOCUMENTS_DIR, file))
                docs.extend(loader.load())
        
        if docs:
            # Fragmentos más grandes (1200) para no perder contexto de precios
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
            chunks = splitter.split_documents(docs)
            with index_lock:
                vector_db = FAISS.from_documents(chunks, embeddings)
                vector_db.save_local(INDEX_FILE)
            logging.info("¡Base de datos de ventas actualizada y lista!")
        else:
            logging.error("ATENCIÓN: No hay archivos PDF en la carpeta /documents")
    except Exception as e:
        logging.error(f"Error crítico en indexación: {e}")

# ---------------- ENDPOINTS ----------------

@app.post("/vapi-webhook")
async def vapi_webhook(request: Request):
    data = await request.json()
    try:
        # Vapi envía la pregunta en 'query' como configuramos
        message_data = data.get("message", {})
        tool_calls = message_data.get("toolCalls", [])
        
        if tool_calls:
            tool_call = tool_calls[0]
            args = tool_call.get("function", {}).get("arguments", {})
            user_query = args.get("query") or args.get("message") or ""
            
            if user_query:
                respuesta = process_message("vapi_user", user_query)
                # Devolvemos el resultado en 'result' como configuramos
                return {"results": [{"toolCallId": tool_call.get("id"), "result": respuesta}]}
    except Exception as e:
        logging.error(f"Error en Webhook Vapi: {e}")
    return {"ok": True}

@app.on_event("startup")
async def startup():
    init_db()
    # Ejecutar indexación al arrancar
    threading.Thread(target=build_index, daemon=True).start()

@app.get("/")
def root(): return {"status": "Sales Agent Online & Multilingual"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
