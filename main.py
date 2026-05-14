import os
import uvicorn
import sqlite3
import logging
import threading
import json

from fastapi import FastAPI, Request
from dotenv import load_dotenv

# --- LIBRERÍAS DE OPENAI (Detectan OPENAI_API_KEY por defecto) ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langdetect import detect, DetectorFactory

# Estabilidad para detección de idiomas
DetectorFactory.seed = 0
load_dotenv()

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI(title="Hotel AI Concierge - Standard Config")

# --- CONFIGURACIÓN DE RUTAS ---
DOCUMENTS_DIR = "documents"
DATABASE = "hotel_memory.db"

# --- ESTRATEGIA DE ACENTOS (Hospitalidad) ---
COUNTRY_ADAPTATION = {
    "ar": {"acento": "argentino", "gerga": "Usa 'con gusto', 'che', 'vos'. Sé servicial pero elegante."},
    "mx": {"acento": "mexicano", "gerga": "Usa 'mandé', 'con mucho gusto', 'ahorita'. Sé muy atento."},
    "es": {"acento": "español de España", "gerga": "Usa 'vale', 'dígame', 'estancia'. Sé eficiente y cortés."},
    "default": {"acento": "neutro", "gerga": "Usa un español de hospitalidad estándar y formal."}
}

vector_db = None
index_lock = threading.Lock()

# ---------------- BASE DE DATOS (Memoria del Huésped) ----------------
def init_db():
    # check_same_thread=False permite que múltiples hilos accedan a la DB en FastAPI
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hotel_convs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            user_id TEXT, role TEXT, message TEXT, 
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_message(user_id, role, message):
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO hotel_convs (user_id, role, message) VALUES (?, ?, ?)", (user_id, role, message))
    conn.commit()
    conn.close()

def get_history(user_id, limit=4):
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT role, message FROM hotel_convs WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    rows.reverse()
    return [HumanMessage(content=m) if r == "user" else AIMessage(content=m) for r, m in rows]

# ---------------- MOTOR OPENAI ----------------
def process_hotel_message(user_id, message, country_code="default"):
    try:
        # 1. Detección de idioma/acento
        lang = country_code if country_code != "default" else "es"
        try: 
            if country_code == "default": lang = detect(message)
        except: lang = "es"

        style = COUNTRY_ADAPTATION.get(lang, COUNTRY_ADAPTATION["default"])

        # 2. RAG: Búsqueda de información en los documentos subidos
        contexto = ""
        if vector_db is not None:
            with index_lock:
                docs = vector_db.similarity_search(message, k=2)
                contexto = "\n".join([d.page_content for d in docs])

        # 3. LLM: Se inicializa y busca automáticamente os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
            max_tokens=120
        )

        system_prompt = f"""
        PERSONALIDAD: Eres el Asistente de Recepción de nuestro Hotel.
        TONO: Acento {style['acento']}. {style['gerga']} Sé extremadamente educado.
        
        TAREA: Responde preguntas sobre servicios, horarios y reservas usando el CONTEXTO proporcionado.
        REGLA DE ORO: Máximo 25 palabras. Si no tienes la información, ofrece pasar la consulta a un humano.

        INFORMACIÓN DEL HOTEL (CONTEXTO):
        {contexto if contexto else "Bienvenido a nuestro hotel, ¿en qué puedo asistirle hoy?"}
        """

        history = get_history(user_id)
        messages = [SystemMessage(content=system_prompt), *history, HumanMessage(content=message)]
        
        response = llm.invoke(messages)
        
        # Persistencia de la charla
        save_message(user_id, "user", message)
        save_message(user_id, "assistant", response.content)
        
        return response.content

    except Exception as e:
        logging.error(f"Error en proceso Hotel: {e}")
        return "Bienvenido a recepción. Hubo un pequeño error, pero dígame, ¿cómo puedo ayudarle?"

# ---------------- INDEXACIÓN DE DOCUMENTOS ----------------
def build_index():
    global vector_db
    try:
        # OpenAIEmbeddings también busca automáticamente OPENAI_API_KEY
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        if not os.path.exists(DOCUMENTS_DIR): os.makedirs(DOCUMENTS_DIR)
        
        all_docs = []
        for file in os.listdir(DOCUMENTS_DIR):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(DOCUMENTS_DIR, file))
                all_docs.extend(loader.load())
        
        if all_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(all_docs)
            with index_lock:
                vector_db = FAISS.from_documents(chunks, embeddings)
                logging.info("--- DOCUMENTOS INDEXADOS CON ÉXITO ---")
        else:
            logging.info("No se encontraron PDFs para indexar en /documents")

    except Exception as e:
        logging.error(f"Error en la indexación: {e}")

# ---------------- WEBHOOK PARA VAPI ----------------
@app.post("/vapi-webhook")
async def vapi_webhook(request: Request):
    try:
        data = await request.json()
        message_data = data.get("message", {})

        if "toolCalls" in message_data:
            tc = message_data["toolCalls"][0]
            func = tc.get("function", {})
            args = func.get("arguments", {})
            
            # Limpieza de argumentos JSON
            if isinstance(args, str):
                try: args = json.loads(args)
                except: args = {"query": args}
            
            query = args.get("query", "")
            customer_info = message_data.get("customer", {})
            country = customer_info.get("country", "default").lower()

            respuesta = process_hotel_message("vapi_hotel_user", query, country_code=country)
            
            return {
                "results": [
                    {
                        "toolCallId": tc.get("id"),
                        "result": respuesta 
                    }
                ]
            }
    except Exception as e:
        logging.error(f"Error Webhook: {e}")
        return {"error": str(e)}
    
    return {"ok": True}

@app.on_event("startup")
async def startup():
    init_db()
    # Ejecución de la carga de documentos en hilo separado
    threading.Thread(target=build_index, daemon=True).start()

@app.get("/")
def health_check():
    return {"status": "Online", "engine": "OpenAI GPT-4o-mini"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
