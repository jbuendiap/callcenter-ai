import os
import uvicorn
import sqlite3
import logging
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

from langdetect import detect

# ---------------- CONFIG ----------------

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AI Call Center")

INDEX_FILE = "hotel_faiss_index"
DOCUMENTS_DIR = os.path.join(os.getcwd(), "documents")
DATABASE = "memory.db"

vector_db = None
index_lock = threading.Lock()

# ---------------- MODELS ----------------

class Message(BaseModel):
    user_id: str
    message: str

# ---------------- DATABASE ----------------

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

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO conversations (user_id, role, message)
    VALUES (?, ?, ?)
    """, (user_id, role, message))

    conn.commit()
    conn.close()


def get_history(user_id, limit=10):

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT role, message
    FROM conversations
    WHERE user_id=?
    ORDER BY id DESC
    LIMIT ?
    """, (user_id, limit))

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

# ---------------- LEAD CLASSIFICATION ----------------

def classify_lead(message: str):

    msg = message.lower()

    hot_keywords = [
        "reservar","reserva","disponibilidad",
        "book","booking","reserve","confirmar"
    ]

    warm_keywords = [
        "precio","tarifa","cuanto cuesta",
        "cost","rate","price"
    ]

    for word in hot_keywords:
        if word in msg:
            return "lead_caliente"

    for word in warm_keywords:
        if word in msg:
            return "lead_interesado"

    return "lead_frio"

# ---------------- BOOKING INTENT ----------------

def detect_booking_intent(message: str):

    msg = message.lower()

    booking_keywords = [
        "quiero reservar",
        "reservar",
        "hacer reserva",
        "confirmar reserva",
        "confirmar",
        "book",
        "booking",
        "reserve",
        "confirm booking"
    ]

    for word in booking_keywords:
        if word in msg:
            return True

    return False

# ---------------- INTENT DETECTION ----------------

def detect_intent(message: str):

    msg = message.lower()

    intents = {

        "saludo": [
            "hola","hello","hi",
            "buenos dias","good morning"
        ],

        "precio": [
            "precio","tarifa",
            "cuanto cuesta","price","rate"
        ],

        "disponibilidad": [
            "disponibilidad",
            "available","availability"
        ],

        "reserva": [
            "reservar","reserva",
            "book","booking"
        ],

        "objecion": [
            "caro","expensive",
            "muy caro","too expensive"
        ],

        "comparacion": [
            "mejor que",
            "difference",
            "compare"
        ],

        "cliente_listo": [
            "quiero reservar",
            "confirmar reserva",
            "book now"
        ],

        "despedida": [
            "gracias",
            "thank you",
            "bye"
        ]
    }

    for intent, keywords in intents.items():

        for word in keywords:

            if word in msg:
                return intent

    return "informacion"

# ---------------- LANGUAGE DETECTION ----------------

def detect_language(text):

    try:
        return detect(text)
    except:
        return "es"

# ---------------- VECTOR DATABASE ----------------

def build_index():

    global vector_db

    try:

        embeddings = OpenAIEmbeddings()

        if os.path.exists(INDEX_FILE):

            logging.info("Loading FAISS index")

            vector_db = FAISS.load_local(
                INDEX_FILE,
                embeddings,
                allow_dangerous_deserialization=True
            )

        else:

            logging.info("Building FAISS index")

            docs = []

            for file in os.listdir(DOCUMENTS_DIR):

                if file.endswith(".pdf"):

                    loader = PyPDFLoader(
                        os.path.join(DOCUMENTS_DIR, file)
                    )

                    docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(docs)

            vector_db = FAISS.from_documents(
                chunks,
                embeddings
            )

            vector_db.save_local(INDEX_FILE)

            logging.info("Index created")

    except Exception as e:

        logging.error(f"Index error: {e}")
        vector_db = None

# ---------------- STARTUP ----------------

@app.on_event("startup")
async def startup_event():

    init_db()

    threading.Thread(
        target=build_index,
        daemon=True
    ).start()

# ---------------- CHAT ENDPOINT ----------------

@app.post("/chat")
async def chat(data: Message):

    if vector_db is None:

        return {
            "response": "Estoy cargando la información. Intente nuevamente en unos segundos."
        }

    try:

        language = detect_language(data.message)

        lead_type = classify_lead(data.message)

        intent = detect_intent(data.message)

        booking_intent = detect_booking_intent(data.message)

        save_message(data.user_id, "user", data.message)

        history = get_history(data.user_id)

        with index_lock:

            docs = vector_db.similarity_search(
                data.message,
                k=5
            )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4
        )

        system_prompt = f"""

Responde utilizando únicamente la información disponible en los documentos.

Idioma del usuario: {language}

Intención del cliente detectada: {intent}

Tipo de lead: {lead_type}

Si no encuentras la información en los documentos responde exactamente:

"Esta información la consultaré y le responderé en la brevedad."

Información disponible:

{context}

"""

        if booking_intent:

            system_prompt += """

El cliente parece listo para realizar una reserva.

Debes ayudarle a completar la reserva solicitando:

- Nombre completo
- Fecha de llegada
- Fecha de salida
- Cantidad de personas
- Correo electrónico o teléfono

Guía al cliente para completar la reserva.

"""

        messages = [
            SystemMessage(content=system_prompt)
        ]

        messages.extend(history)

        messages.append(
            HumanMessage(content=data.message)
        )

        response = llm.invoke(messages)

        save_message(
            data.user_id,
            "assistant",
            response.content
        )

        return {

            "response": response.content,
            "lead_type": lead_type,
            "intent": intent,
            "booking_intent": booking_intent,
            "language": language

        }

    except Exception as e:

        logging.error(f"Error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Error procesando solicitud"
        )

# ---------------- UPDATE INDEX ----------------

@app.post("/update-index")
async def update_index():

    threading.Thread(
        target=build_index,
        daemon=True
    ).start()

    return {
        "message": "Index rebuilding"
    }

# ---------------- HISTORY ----------------

@app.get("/history/{user_id}")
async def history(user_id: str):

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""

    SELECT role, message, timestamp
    FROM conversations
    WHERE user_id=?
    ORDER BY id DESC
    LIMIT 20

    """, (user_id,))

    rows = cursor.fetchall()

    conn.close()

    return rows

# ---------------- MAIN ----------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
