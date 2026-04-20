import os
import uvicorn
import sqlite3
import logging
import threading
import shutil

from fastapi import FastAPI, Request
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langdetect import detect, DetectorFactory

# Estabilidad para detección de idiomas
DetectorFactory.seed = 0

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI(title="AI Global Sales Agent - Claude & RAG Optimized")

INDEX_FILE = "sales_faiss_index"
DOCUMENTS_DIR = os.path.join(os.getcwd(), "documents")
DATABASE = "memory.db"

vector_db = None
index_lock = threading.Lock()

# ---------------- BASE DE DATOS (Memoria de Usuario) ----------------
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

# ---------------- MOTOR DE INTELIGENCIA CLAUDE 3.5 ----------------
def process_message(user_id, message):
    try:
        # Detección de idioma para responder en el mismo tono
        try:
            lang_code = detect(message)
        except:
            lang_code = "es"

        save_message(user_id, "user", message)
        history = get_history(user_id)

        # BÚSQUEDA SEMÁNTICA EN TUS DOCUMENTOS (Forex/Hoteles)
        contexto_encontrado = ""
        if vector_db is not None:
            with index_lock:
                # Buscamos los 5 fragmentos más relevantes
                docs_with_score = vector_db.similarity_search_with_relevance_scores(message, k=5)
                
                # Filtramos para que solo use info realmente útil (>0.5 de relevancia)
                contexto_encontrado = "\n\n".join([
                    f"[INFO DE APOYO]: {d.page_content}" 
                    for d, score in docs_with_score if score > 0.5
                ])

        # CONFIGURACIÓN DE CLAUDE (El cerebro persuasivo)
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20240620", 
            temperature=0.4, # Un poco de calor para que suene humano, no robot
            max_tokens=200   # Respuestas breves para evitar que el cliente se aburra
        )
        
        system_rules = f"""
        PERSONALIDAD: Eres Juan Esteban, un Consultor de Ventas Senior. No eres un bot, eres un experto.
        TU TONO: Seguro, carismático y empático. Habla con naturalidad (usa "Mire", "Le explico", "Entiendo su punto").

        TU FUENTE DE DATOS: Utiliza el 'CONTEXTO RECUPERADO' para tus respuestas.
        SI NO HAY INFORMACIÓN: No inventes datos técnicos. Usa tu carisma para despertar interés y ofrece hablar con un director.

        REGLAS DE CIERRE:
        1. Si el cliente duda por seguridad, usa la psicología de los documentos para dar tranquilidad.
        2. Mantén las respuestas de máximo 2 oraciones. Es una llamada, no un correo.
        3. Siempre responde en el idioma detectado: {lang_code}.

        CONTEXTO RECUPERADO (DATOS REALES):
        {contexto_encontrado if contexto_encontrado else "Usa técnicas de apertura de ventas generales."}
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
        logging.error(f"Error en el motor: {e}")
        return "Disculpe, la línea está un poco inestable. ¿Podría repetirme eso último?"

# ---------------- INDEXACIÓN AUTOMÁTICA (Tus PDFs) ----------------
def build_index():
    global vector_db
    try:
        # Para los embeddings seguimos usando OpenAI (es más barato y compatible)
        embeddings = OpenAIEmbeddings()
        
        if os.path.exists(INDEX_FILE):
            shutil.rmtree(INDEX_FILE)

        if not os.path.exists(DOCUMENTS_DIR): 
            os.makedirs(DOCUMENTS_DIR)
        
        all_docs = []
        for file in os.listdir(DOCUMENTS_DIR):
            if file.endswith(".pdf"):
                path = os.path.join(DOCUMENTS_DIR, file)
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())
        
        if all_docs:
            # Fragmentos pequeños para que la IA los procese rápido
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(all_docs)
            with index_lock:
                vector_db = FAISS.from_documents(chunks, embeddings)
                vector_db.save_local(INDEX_FILE)
            logging.info("¡Documentos sincronizados y listos para vender!")
    except Exception as e:
        logging.error(f"Error en indexación: {e}")

# ---------------- CONEXIÓN CON VAPI ----------------
@app.post("/vapi-webhook")
async def vapi_webhook(request: Request):
    data = await request.json()
    try:
        # Vapi envía el texto del usuario aquí
        tool_calls = data.get("message", {}).get("toolCalls", [])
        if tool_calls:
            tc = tool_calls[0]
            args = tc.get("function", {}).get("arguments", {})
            user_query = args.get("query") or args.get("message") or ""
            
            if user_query:
                respuesta = process_message("vapi_user", user_query)
                return {"results": [{"toolCallId": tc.get("id"), "result": respuesta}]}
    except Exception as e:
        logging.error(f"Error Webhook: {e}")
    
    return {"ok": True}

@app.on_event("startup")
async def startup():
    init_db()
    # Ejecutamos la indexación en un hilo separado para no bloquear el inicio
    threading.Thread(target=build_index, daemon=True).start()

@app.get("/")
def root(): 
    return {"status": "Vendedor Online", "engine": "Claude 3.5 Sonnet"}

if __name__ == "__main__":
    # Railway asigna el puerto automáticamente
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
