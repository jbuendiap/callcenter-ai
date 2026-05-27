import os
import uvicorn
import sqlite3
import logging
import threading
import json

from fastapi import FastAPI, Request
from dotenv import load_dotenv

# --- LIBRERÍAS OPENAI ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langdetect import detect, DetectorFactory

# ---------------- CONFIGURACIÓN ----------------

DetectorFactory.seed = 0
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI(title="Hotel AI Concierge")

DOCUMENTS_DIR = "documents"
DATABASE = "hotel_memory.db"

# ---------------- ADAPTACIÓN DE TONO ----------------

COUNTRY_ADAPTATION = {
    "ar": {
        "acento": "argentino",
        "gerga": "Usa 'con gusto', 'che', 'vos'."
    },
    "mx": {
        "acento": "mexicano",
        "gerga": "Usa 'mandé', 'con mucho gusto', 'ahorita'."
    },
    "es": {
        "acento": "español",
        "gerga": "Usa 'vale', 'dígame', 'estancia'."
    },
    "default": {
        "acento": "neutro",
        "gerga": "Usa español formal y elegante."
    }
}

vector_db = None
index_lock = threading.Lock()

# ---------------- BASE DE DATOS ----------------

def init_db():
    conn = sqlite3.connect(DATABASE, check_same_thread=False)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hotel_convs (
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
    conn = sqlite3.connect(DATABASE, check_same_thread=False)

    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO hotel_convs (user_id, role, message) VALUES (?, ?, ?)",
        (user_id, role, message)
    )

    conn.commit()
    conn.close()


def get_history(user_id, limit=4):
    conn = sqlite3.connect(DATABASE, check_same_thread=False)

    cursor = conn.cursor()

    cursor.execute(
        "SELECT role, message FROM hotel_convs WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, limit)
    )

    rows = cursor.fetchall()

    conn.close()

    rows.reverse()

    return [
        HumanMessage(content=m) if r == "user"
        else AIMessage(content=m)
        for r, m in rows
    ]

# ---------------- IA PRINCIPAL ----------------

def process_hotel_message(user_id, message, country_code="default"):

    try:

        # Detectar idioma
        lang = country_code if country_code != "default" else "es"

        try:
            if country_code == "default":
                lang = detect(message)
        except:
            lang = "es"

        style = COUNTRY_ADAPTATION.get(
            lang,
            COUNTRY_ADAPTATION["default"]
        )

        # Buscar contexto RAG
        contexto = ""

        if vector_db is not None:

            with index_lock:

                docs = vector_db.similarity_search(message, k=2)

                contexto = "\n".join([
                    d.page_content for d in docs
                ])

        # Modelo OpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
            max_tokens=120
        )

        system_prompt = f"""
        Eres el recepcionista virtual de un hotel.

        TONO:
        - Acento {style['acento']}
        - {style['gerga']}

        REGLAS:
        - Máximo 25 palabras.
        - Sé elegante y amable.
        - Usa el contexto del hotel.
        - Si no sabes algo, ofrece pasar con un humano.

        CONTEXTO:
        {contexto if contexto else "Bienvenido al hotel."}
        """

        history = get_history(user_id)

        messages = [
            SystemMessage(content=system_prompt),
            *history,
            HumanMessage(content=message)
        ]

        response = llm.invoke(messages)

        # Guardar conversación
        save_message(user_id, "user", message)
        save_message(user_id, "assistant", response.content)

        return response.content

    except Exception as e:

        logging.error(f"ERROR IA: {e}")

        return "Disculpe, ocurrió un error temporal."

# ---------------- INDEXACIÓN PDFs ----------------

def build_index():

    global vector_db

    try:

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        if not os.path.exists(DOCUMENTS_DIR):
            os.makedirs(DOCUMENTS_DIR)

        all_docs = []

        for file in os.listdir(DOCUMENTS_DIR):

            if file.endswith(".pdf"):

                loader = PyPDFLoader(
                    os.path.join(DOCUMENTS_DIR, file)
                )

                all_docs.extend(loader.load())

        if all_docs:

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(all_docs)

            with index_lock:

                vector_db = FAISS.from_documents(
                    chunks,
                    embeddings
                )

            logging.info("DOCUMENTOS INDEXADOS")

        else:

            logging.info("NO HAY PDFs")

    except Exception as e:

        logging.error(f"ERROR INDEXACIÓN: {e}")

# ---------------- WEBHOOK VAPI ----------------

@app.post("/vapi-webhook")
async def vapi_webhook(request: Request):

    try:

        data = await request.json()

        logging.info(f"DATA RECIBIDA: {data}")

        message_data = data.get("message", {})

        # -------- TOOL CALLS --------

        if "toolCalls" in message_data:

            tc = message_data["toolCalls"][0]

            func = tc.get("function", {})

            args = func.get("arguments", {})

            if isinstance(args, str):

                try:
                    args = json.loads(args)
                except:
                    args = {"query": args}

            query = args.get("query", "")

            respuesta = process_hotel_message(
                "vapi_hotel_user",
                query
            )

            return {
                "results": [
                    {
                        "toolCallId": tc.get("id"),
                        "result": respuesta
                    }
                ]
            }

        # -------- MENSAJE NORMAL --------

        transcript = message_data.get("transcript", "")

        if transcript:

            respuesta = process_hotel_message(
                "vapi_hotel_user",
                transcript
            )

            return {
                "response": respuesta
            }

        return {"ok": True}

    except Exception as e:

        logging.error(f"ERROR WEBHOOK: {e}")

        return {
            "error": str(e)
        }

# ---------------- STARTUP ----------------

@app.on_event("startup")
async def startup():

    init_db()

    threading.Thread(
        target=build_index,
        daemon=True
    ).start()

# ---------------- HEALTH CHECK ----------------

@app.get("/")
def health_check():

    return {
        "status": "Online",
        "engine": "OpenAI GPT-4o-mini"
    }

# ---------------- MAIN ----------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
