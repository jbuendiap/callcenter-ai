import os
import uvicorn
import sqlite3
import logging
import threading
import json
import asyncio

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langdetect import detect


# ---------------- CONFIGURACIÓN ----------------

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# VARIABLE PARA ACTIVAR / DESACTIVAR BOT
BOT_ACTIVE = os.getenv("BOT_ACTIVE", "true")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="AI Call Center Pro - Lead Scoring & Emotion AI")

INDEX_FILE = "hotel_faiss_index"
DOCUMENTS_DIR = os.path.join(os.getcwd(), "documents")
DATABASE = "memory.db"

vector_db = None
index_lock = threading.Lock()


class Message(BaseModel):
    user_id: str
    message: str


# ---------------- BASE DE DATOS ----------------

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
        score INTEGER DEFAULT 0
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

    cursor.execute(
        "INSERT INTO conversations (user_id, role, message) VALUES (?, ?, ?)",
        (user_id, role, message),
    )

    conn.commit()
    conn.close()


def get_history(user_id, limit=8):

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT role, message FROM conversations WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    )

    rows = cursor.fetchall()
    conn.close()

    rows.reverse()

    history = []

    for role, message in rows:

        if role == "user":
            history.append(HumanMessage(content=message))
        else:
            history.append(AIMessage(content=message))

    return history


# ---------------- ANALÍTICA ----------------

def analyze_customer_behavior(message):

    llm_analyst = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
    Analiza el siguiente mensaje.

    Mensaje: "{message}"

    Responde JSON:

    {{"intencion":"valor","emocion":"valor"}}
    """

    try:

        response = llm_analyst.invoke(prompt)
        data = json.loads(response.content)

        return data.get("intencion","informacion"), data.get("emocion","neutral")

    except Exception as e:

        logging.warning(e)
        return "informacion","neutral"


def calculate_points(intent, emotion):

    points = 5

    intent_points = {
        "precio":15,
        "reserva":40,
        "disponibilidad":20,
        "objecion":-10,
        "comparacion":5,
        "saludo":2
    }

    emotion_points = {
        "decision_compra":30,
        "frustracion":-10,
        "satisfaccion":15,
        "duda":-5
    }

    points += intent_points.get(intent,0)
    points += emotion_points.get(emotion,0)

    return points


def update_lead_data(user_id, points, language, intent, emotion):

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO lead_scores (user_id, score) VALUES(?, ?)
    ON CONFLICT(user_id) DO UPDATE SET score = score + excluded.score
    """,(user_id,points))

    cursor.execute("""
    INSERT OR REPLACE INTO customer_profile (user_id,language,last_intent,last_emotion)
    VALUES (?,?,?,?)
    """,(user_id,language,intent,emotion))

    conn.commit()

    cursor.execute("SELECT score FROM lead_scores WHERE user_id=?",(user_id,))
    score = cursor.fetchone()[0]

    conn.close()

    return score


# ---------------- RAG ----------------

def build_index():

    global vector_db

    try:

        embeddings = OpenAIEmbeddings()

        if os.path.exists(INDEX_FILE):

            vector_db = FAISS.load_local(
                INDEX_FILE,
                embeddings,
                allow_dangerous_deserialization=True
            )

        else:

            docs = []

            if not os.path.exists(DOCUMENTS_DIR):
                os.makedirs(DOCUMENTS_DIR)

            for file in os.listdir(DOCUMENTS_DIR):

                if file.endswith(".pdf"):

                    loader = PyPDFLoader(os.path.join(DOCUMENTS_DIR,file))
                    docs.extend(loader.load())

            if docs:

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100
                )

                chunks = splitter.split_documents(docs)

                vector_db = FAISS.from_documents(chunks,embeddings)

                vector_db.save_local(INDEX_FILE)

        logging.info("Base de conocimiento cargada")

    except Exception as e:

        logging.error(e)


# ---------------- MOTOR IA ----------------

def process_message(user_id,message):

    # BOT APAGADO
    if BOT_ACTIVE.lower() != "true":
        return "El asistente está temporalmente desactivado."

    if vector_db is None:
        return "Inicializando conocimiento..."

    intent,emotion = analyze_customer_behavior(message)

    language = detect(message) if len(message)>3 else "es"

    points = calculate_points(intent,emotion)

    total_score = update_lead_data(user_id,points,language,intent,emotion)

    save_message(user_id,"user",message)

    history = get_history(user_id)

    with index_lock:
        docs = vector_db.similarity_search(message,k=4)

    contexto = "\n\n".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.4)

    system_rules=f"""
    Eres experto en ventas hoteleras.

    idioma:{language}
    intent:{intent}
    emocion:{emotion}
    score:{total_score}

    Usa el contexto para responder.

    CONTEXTO:
    {contexto}
    """

    messages=[
        SystemMessage(content=system_rules),
        *history,
        HumanMessage(content=message)
    ]

    response=llm.invoke(messages)

    save_message(user_id,"assistant",response.content)

    return response.content


# ---------------- API CHAT ----------------

@app.post("/chat")
async def chat(data:Message):

    try:

        response = process_message(data.user_id,data.message)

        return {"response":response}

    except Exception as e:

        logging.error(e)
        raise HTTPException(500,"error interno")


# ---------------- TELEGRAM BOT ----------------

async def telegram_message(update:Update,context:ContextTypes.DEFAULT_TYPE):

    if not update.message or not update.message.text:
        return

    user_id = str(update.message.from_user.id)
    message = update.message.text

    reply = process_message(user_id,message)

    await update.message.reply_text(reply)


async def start_telegram_bot():

    if not TELEGRAM_TOKEN:

        logging.warning("No TELEGRAM_BOT_TOKEN definido")
        return

    bot = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    bot.add_handler(MessageHandler(filters.TEXT,telegram_message))

    logging.info("Telegram bot iniciado")

    await bot.run_polling()


# ---------------- STARTUP ----------------

@app.on_event("startup")
async def startup():

    init_db()

    threading.Thread(target=build_index,daemon=True).start()

    asyncio.create_task(start_telegram_bot())


# ---------------- RUN ----------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT",8000))

    uvicorn.run(app,host="0.0.0.0",port=port)
