import os
import uvicorn
import sqlite3
import logging
import threading
import shutil

from fastapi import FastAPI, Request
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langdetect import detect, DetectorFactory

# Estabilidad para idiomas
DetectorFactory.seed = 0

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI(title="AI Hotel Elite Sales Agent - RAG Optimized")

INDEX_FILE = "hotel_faiss_index"
DOCUMENTS_DIR = os.path.join(os.getcwd(), "documents")
DATABASE = "memory.db"

vector_db = None
index_lock = threading.Lock()

# ---------------- BASE DE DATOS ----------------
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, role TEXT, message TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
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

# ---------------- MOTOR IA MEJORADO ----------------
def process_message(user_id, message):
    try:
        # Detección de idioma
        try:
            lang_code = detect(message)
        except:
            lang_code = "es"

        lang_map = {"es": "Español", "en": "Inglés", "ru": "Ruso", "fr": "Francés"}
        idioma_destino = lang_map.get(lang_code, "Español")

        save_message(user_id, "user", message)
        history = get_history(user_id)

        # BÚSQUEDA SEMÁNTICA MEJORADA
        contexto_encontrado = ""
        if vector_db is not None:
            with index_lock:
                # Buscamos con puntuación de relevancia para filtrar basura
                docs_with_score = vector_db.similarity_search_with_relevance_scores(message, k=8)
                
                # Solo usamos documentos con una relevancia aceptable (> 0.6)
                contexto_encontrado = "\n\n".join([
                    f"[FRAGMENTO DE DOCUMENTO: {d.metadata.get('source', 'Desconocido')}]:\n{d.page_content}" 
                    for d, score in docs_with_score if score > 0.6
                ])

        if not contexto_encontrado:
            contexto_encontrado = "No hay datos específicos en los manuales. Responde que validarás con un humano."

        # Modelo más preciso (Temperature 0 para evitar inventar datos)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        system_rules = f"""
        INSTRUCCIÓN PRIMARIA: Eres un Gerente de Ventas de Lujo. 
        TU ÚNICA FUENTE DE VERDAD ES EL 'CONTEXTO RECUPERADO'. Si la información no está ahí, di que no la tienes.

        REGLAS DE OPERACIÓN:
        1. IDIOMA: Responde siempre en {idioma_destino}.
        2. PRECIOS: Busca símbolos de '$' o 'USD' en el contexto. Si no hay precios en el contexto, NO LOS INVENTES.
        3. QUEJAS: Aplica estrictamente las técnicas de 'manejo_objeciones.pdf' presentes en el contexto.
        4. VERIFICACIÓN: Antes de responder, verifica si el dato está escrito en el CONTEXTO abajo.

        CONTEXTO RECUPERADO (DATOS REALES):
        {contexto_encontrado}
        """

        messages = [SystemMessage(content=system_rules), *history, HumanMessage(content=message)]
        response = llm.invoke(messages)
        save_message(user_id, "assistant", response.content)
        return response.content

    except Exception as e:
        logging.error(f"Error: {e}")
        return "Disculpe, por favor repita su consulta."

# ---------------- INDEXACIÓN CON METADATOS ----------------
def build_index():
    global vector_db
    try:
        embeddings = OpenAIEmbeddings()
        
        # Limpieza de caché para asegurar que lea los últimos archivos subidos
        if os.path.exists(INDEX_FILE):
            shutil.rmtree(INDEX_FILE)

        if not os.path.exists(DOCUMENTS_DIR): 
            os.makedirs(DOCUMENTS_DIR)
        
        all_docs = []
        for file in os.listdir(DOCUMENTS_DIR):
            if file.endswith(".pdf"):
                path = os.path.join(DOCUMENTS_DIR, file)
                loader = PyPDFLoader(path)
                # Cargamos con metadatos de origen
                all_docs.extend(loader.load())
        
        if all_docs:
            # Fragmentos optimizados para no romper tablas de precios
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
            chunks = splitter.split_documents(all_docs)
            with index_lock:
                vector_db = FAISS.from_documents(chunks, embeddings)
                vector_db.save_local(INDEX_FILE)
            logging.info("¡Base de datos vectorial sincronizada!")
    except Exception as e:
        logging.error(f"Error en indexación: {e}")

# ---------------- ENDPOINTS VAPI ----------------
@app.post("/vapi-webhook")
async def vapi_webhook(request: Request):
    data = await request.json()
    try:
        tool_calls = data.get("message", {}).get("toolCalls", [])
        if tool_calls:
            tc = tool_calls[0]
            args = tc.get("function", {}).get("arguments", {})
            user_query = args.get("query") or args.get("message") or ""
            if user_query:
                respuesta = process_message("vapi_user", user_query)
                return {"results": [{"toolCallId": tc.get("id"), "result": respuesta}]}
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
