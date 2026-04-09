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

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS lead_scores (
        user_id TEXT PRIMARY KEY,
        score INTEGER
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


# ---------------- CUSTOMER PROFILE ----------------

def update_customer_profile(user_id, language, intent, emotion):

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR REPLACE INTO customer_profile (user_id, language, last_intent, last_emotion)
    VALUES (?, ?, ?, ?)
    """, (user_id, language, intent, emotion))

    conn.commit()
    conn.close()


# ---------------- LEAD SCORING ----------------

def get_score(user_id):

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("SELECT score FROM lead_scores WHERE user_id=?", (user_id,))
    result = cursor.fetchone()

    conn.close()

    if result:
        return result[0]

    return 0


def update_score(user_id, points):

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    current_score = get_score(user_id)

    new_score = current_score + points

    cursor.execute("""
    INSERT OR REPLACE INTO lead_scores (user_id, score)
    VALUES (?, ?)
    """, (user_id, new_score))

    conn.commit()
    conn.close()

    return new_score


def calculate_score(message):

    msg = message.lower()

    score = 0

    if "precio" in msg or "price" in msg:
        score += 10

    if "disponibilidad" in msg or "available" in msg:
        score += 20

    if "reservar" in msg or "booking" in msg:
        score += 40

    if "formas de pago" in msg or "payment" in msg:
        score += 20

    return score


# ---------------- LEAD CLASSIFICATION ----------------

def classify_lead(score):

    if score >= 80:
        return "lead_listo_para_comprar"

    if score >= 50:
        return "lead_caliente"

    if score >= 20:
        return "lead_interesado"

    return "lead_frio"


# ---------------- INTENT ----------------

def detect_intent(message):

    msg = message.lower()

    intents = {
        "saludo": ["hola", "hello", "hi"],
        "precio": ["precio", "price"],
        "disponibilidad": ["disponibilidad", "available"],
        "reserva": ["reservar", "booking"],
        "objecion": ["caro", "expensive"],
        "comparacion": ["mejor que", "compare"],
        "cliente_listo": ["quiero reservar", "confirmar reserva"],
        "despedida": ["gracias", "bye"]
    }

    for intent, words in intents.items():
        for word in words:
            if word in msg:
                return intent

    return "informacion"


# ---------------- EMOTION DETECTION ----------------

def detect_emotion(message):

    msg = message.lower()

    if "caro" in msg or "expensive" in msg:
        return "frustracion"

    if "gracias" in msg or "thank" in msg:
        return "satisfaccion"

    if "quiero reservar" in msg or "confirmar" in msg:
        return "decision_compra"

    if "no estoy seguro" in msg or "not sure" in msg:
        return "duda"

    return "neutral"


# ---------------- CLOSE PREDICTION ----------------

def predict_close(score, emotion):

    if score >= 80:
        return "muy_probable"

    if score >= 60 and emotion == "decision_compra":
        return "probable"

    if score >= 40:
        return "posible"

    return "baja_probabilidad"


# ---------------- LANGUAGE ----------------

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


# ---------------- CHAT ----------------

@app.post("/chat")
async def chat(data: Message):

    if vector_db is None:

        return {
            "response": "Estoy cargando la información. Intente nuevamente en unos segundos."
        }

    try:

        language = detect_language(data.message)

        intent = detect_intent(data.message)

        emotion = detect_emotion(data.message)

        booking_intent = "reservar" in data.message.lower()

        points = calculate_score(data.message)

        score = update_score(data.user_id, points)

        lead_type = classify_lead(score)

        close_prediction = predict_close(score, emotion)

        update_customer_profile(
            data.user_id,
            language,
            intent,
            emotion
        )

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

Responde únicamente usando la información de los documentos.

Idioma: {language}

Tipo de lead: {lead_type}

Score del cliente: {score}

Intención: {intent}

Emoción del cliente: {emotion}

Probabilidad de cierre: {close_prediction}

Si no encuentras la información responde exactamente:

"Esta información la consultaré y le responderé en la brevedad."

Información disponible:

{context}

"""

        if score >= 80 or booking_intent:

            system_prompt += """

El cliente parece listo para reservar.

Solicita:

- Nombre completo
- Fecha de llegada
- Fecha de salida
- Cantidad de personas
- Email o teléfono

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
            "score": score,
            "lead_type": lead_type,
            "intent": intent,
            "emotion": emotion,
            "close_prediction": close_prediction,
            "language": language

        }

    except Exception as e:

        logging.error(f"Error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Error procesando solicitud"
        )


# ---------------- REBUILD INDEX ----------------

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
