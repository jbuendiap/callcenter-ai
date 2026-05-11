import os
import uvicorn
import sqlite3
import logging
import threading
import json

from fastapi import FastAPI, Request
from dotenv import load_dotenv

# Librerías de Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langdetect import detect, DetectorFactory

# Estabilidad para detección de idiomas
DetectorFactory.seed = 0
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI(title="Hotel AI Concierge - Gemini Powered")

# --- CONFIGURACIÓN DE RUTAS ---
DOCUMENTS_DIR = "documents"
DATABASE = "hotel_memory.db" # Cambiado para no chocar con Forex

# --- ESTRATEGIA DE ACENTOS PARA HOTELES (Hospitalidad) ---
COUNTRY_ADAPTATION = {
    "ar": {"acento": "argentino", "jerga": "Usa 'con gusto', 'che', 'vos'. Sé servicial pero elegante."},
    "mx": {"acento": "mexicano", "jerga": "Usa 'mandé', 'con mucho gusto', 'ahorita'. Sé muy atento."},
    "es": {"acento": "español de España", "jerga": "Usa 'vale', 'dígame', 'estancia'. Sé eficiente y cortés."},
    "default": {"acento": "neutro", "jerga": "Usa un español de hospitalidad estándar y formal."}
}

vector_db = None
index_lock = threading.Lock()

# ---------------- BASE DE DATOS (Memoria del Huésped) ----------------
def init_db():
    conn = sqlite3.connect(DATABASE)
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
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO hotel_convs (user_id, role, message) VALUES (?, ?, ?)", (user_id, role, message))
    conn.commit()
    conn.close()

def get_history(user_id, limit=4):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, message FROM hotel_convs WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    rows.reverse()
    return [HumanMessage(content=m) if r == "user" else AIMessage(content=m) for r, m in rows]

# ---------------- MOTOR GEMINI (Específico Hotel) ----------------
def process_hotel_message(user_id, message, country_code="default"):
    try:
        lang = country_code if country_code != "default" else "es"
        try: 
            if country_code == "default": lang = detect(message)
        except: lang = "es"

        style = COUNTRY_ADAPTATION.get(lang, COUNTRY_ADAPTATION["default"])

        # RAG: Buscar en los PDFs de la carpeta 'documents' (Hoteles)
        contexto = ""
        if vector_db is not None:
            with index_lock:
                # Busca información relevante en hotel_info.pdf, etc.
                docs = vector_db.similarity_search(message, k=2)
                contexto = "\n".join([d.page_content for d in docs])

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.4, # Menos creativo, más preciso con datos del hotel
            max_output_tokens=120
        )

        system_prompt = f"""
        PERSONALIDAD: Eres el Asistente de Recepción de nuestro Hotel.
        TONO: Acento {style['acento']}. {style['jerga']} Sé extremadamente educado.
        
        TAREA: Responde preguntas sobre servicios, horarios y reservas usando el CONTEXTO.
        REGLA DE ORO: Máximo 25 palabras. Si no sabes algo por el contexto, ofrece ayuda humana.

        INFORMACIÓN DEL HOTEL (CONTEXTO):
        {contexto if contexto else "Ofrece una bienvenida cordial y pregunta en qué puedes ayudar."}
        """

        history = get_history(user_id)
        messages = [SystemMessage(content=system_prompt), *history, HumanMessage(content=message)]
        
        response = llm.invoke(messages)
        save_message(user_id, "user", message)
        save_message(user_id, "assistant", response.content)
        
        return response.content

    except Exception as e:
        logging.error(f"Error en Hotel: {e}")
        return "Bienvenido a recepción. ¿En qué puedo asistirle?"

# ---------------- INDEXACIÓN (Busca tus archivos en /documents) ----------------
def build_index():
    global vector_db
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if not os.path.exists(DOCUMENTS_DIR): os.makedirs(DOCUMENTS_DIR)
        
        all_docs = []
        # Esto leerá automáticamente hotel_info.pdf y los otros que tienes en la captura
        for file in os.listdir(DOCUMENTS_DIR):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(DOCUMENTS_DIR, file))
                all_docs.extend(loader.load())
        
        if all_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(all_docs)
            with index_lock:
                vector_db = FAISS.from_documents(chunks, embeddings)
                logging.info("--- DOCUMENTOS DE HOTELES CARGADOS ---")
    except Exception as e:
        logging.error(f"Error cargando documentos: {e}")

# ---------------- WEBHOOK VAPI ----------------
@app.post("/vapi-webhook")
async def vapi_webhook(request: Request):
    try:
        data = await request.json()
        message_data = data.get("message", {})

        if "toolCalls" in message_data:
            tc = message_data["toolCalls"][0]
            args = tc.get("function", {}).get("arguments", {})
            
            if isinstance(args, str): args = json.loads(args)
            
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
        logging.error(f"Error Webhook Hotel: {e}")
        return {"error": str(e)}
    
    return {"ok": True}

@app.on_event("startup")
async def startup():
    init_db()
    threading.Thread(target=build_index, daemon=True).start()

@app.get("/")
def health_check():
    return {"status": "Recepción Online", "port": "8080"}

if __name__ == "__main__":
    # Asegurando el puerto 8080 para Railway
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
