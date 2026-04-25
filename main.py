import os
import uvicorn
import sqlite3
import logging
import threading
import shutil

from fastapi import FastAPI, Request
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langdetect import detect, DetectorFactory

# Estabilidad para detección de idiomas
DetectorFactory.seed = 0
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI(title="Forex AI Agent - Antigravity Optimized")

# --- CONFIGURACIÓN DE RUTAS (Estructura IDX) ---
INDEX_FILE = "sales_faiss_index"
DOCUMENTS_DIR = "documents"
DATABASE = "memory.db"

# --- MAPEO DE ACENTOS Y PAÍSES (Estrategia de Ventas) ---
COUNTRY_ADAPTATION = {
    "ar": {"acento": "argentino", "modismos": "usá palabras como 'plata', 'che', 'laburo'. Sé directo."},
    "mx": {"acento": "mexicano", "modismos": "usa 'lana', 'platicar', 'ahorita'. Sé muy amable y formal."},
    "es": {"acento": "español de España", "modismos": "usa 'vale', 'ordenador', 'venga'. Sé profesional y rápido."},
    "default": {"acento": "neutro latino", "modismos": "usa español profesional estándar."}
}

vector_db = None
index_lock = threading.Lock()

# ---------------- BASE DE DATOS (Optimizado para Railway) ----------------
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            user_id TEXT, role TEXT, message TEXT, 
            country_code TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# ---------------- MOTOR CLAUDE 3.5 CON ADAPTACIÓN CULTURAL ----------------
def process_message(user_id, message, country_code="default"):
    try:
        # 1. Detección automática si no viene de la API
        if country_code == "default":
            try: lang_code = detect(message)
            except: lang_code = "es"
        else:
            lang_code = country_code

        adaptation = COUNTRY_ADAPTATION.get(lang_code, COUNTRY_ADAPTATION["default"])

        # 2. RAG: Búsqueda de objeciones de Forex / Info de Hotel
        contexto_encontrado = ""
        if vector_db is not None:
            with index_lock:
                docs = vector_db.similarity_search(message, k=3)
                contexto_encontrado = "\n".join([d.page_content for d in docs])

        # 3. Configuración del LLM (Claude 3.5 Sonnet)
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0.5, # Balance entre creatividad y precisión
            max_tokens=150   # Más corto para menor latencia en Vapi
        )

        system_rules = f"""
        IDENTIDAD: Eres Juan Esteban, Consultor Senior de Inversiones (Forex).
        ADAPTACIÓN CULTURAL: Tu acento es {adaptation['acento']}. {adaptation['modismos']}
        
        REGLAS DE ORO:
        1. No parezcas un bot. Si el cliente interrumpe, sé fluido.
        2. Usa el CONTEXTO para rebatir objeciones de riesgo o seguridad.
        3. Respuestas de MÁXIMO 20 palabras. Queremos diálogo, no monólogo.
        
        CONTEXTO DE APOYO:
        {contexto_encontrado}
        """

        history = get_history(user_id)
        messages = [SystemMessage(content=system_rules), *history, HumanMessage(content=message)]
        
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        logging.error(f"Error: {e}")
        return "Le escucho un poco entrecortado, ¿me decía?"

# ---------------- WEBHOOK PARA VAPI (Integración Directa) ----------------
@app.post("/vapi-webhook")
async def vapi_webhook(request: Request):
    data = await request.json()
    # Extraemos el país si Vapi nos da el número o el dato
    customer_country = data.get("message", {}).get("customer", {}).get("country", "default")
    
    # Manejo de Tool Calls de Vapi
    message_content = ""
    if "message" in data and "toolCalls" in data["message"]:
        tc = data["message"]["toolCalls"][0]
        query = tc.get("function", {}).get("arguments", {}).get("query", "")
        
        respuesta = process_message("vapi_user", query, country_code=customer_country)
        
        return {
            "results": [{
                "toolCallId": tc.get("id"),
                "result": respuesta
            }]
        }
    return {"ok": True}

# ... (Resto de funciones de DB y build_index se mantienen similares) ...
