import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import threading
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="AI Call Center")

class Message(BaseModel):
    user_id: str
    message: str

INDEX_FILE = "hotel_faiss_index"
DOCUMENTS_DIR = os.path.join(os.getcwd(), "documents")

vector_db = None
index_lock = threading.Lock()

conversation_histories: Dict[str, List] = {}

MAX_HISTORY = 10

# -------------------------
# Crear índice FAISS
# -------------------------

def build_index():

    global vector_db

    try:

        embeddings = OpenAIEmbeddings()

        if os.path.exists(INDEX_FILE):

            logging.info("Cargando índice FAISS existente")

            vector_db = FAISS.load_local(
                INDEX_FILE,
                embeddings,
                allow_dangerous_deserialization=True
            )

        else:

            logging.info("Creando índice FAISS desde PDFs")

            docs = []

            for filename in os.listdir(DOCUMENTS_DIR):

                if filename.lower().endswith(".pdf"):

                    loader = PyPDFLoader(
                        os.path.join(DOCUMENTS_DIR, filename)
                    )

                    docs.extend(loader.load())

            if not docs:

                logging.warning("No se encontraron PDFs")
                vector_db = None
                return

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(docs)

            vector_db = FAISS.from_documents(chunks, embeddings)

            vector_db.save_local(INDEX_FILE)

            logging.info("Índice FAISS creado")

    except Exception as e:

        logging.error(f"Error creando índice: {e}")
        vector_db = None


@app.on_event("startup")
async def startup_event():

    threading.Thread(
        target=build_index,
        daemon=True
    ).start()


# -------------------------
# Detectar idioma
# -------------------------

def detect_language(message):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    messages = [

        SystemMessage(content="""
Detect the language of the message.

Return ONLY the language name.

Example:
English
Spanish
French
German
Portuguese
"""),

        HumanMessage(content=message)
    ]

    response = llm.invoke(messages)

    return response.content.strip()


# -------------------------
# Detectar intención
# -------------------------

def detect_intent(message):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    messages = [

        SystemMessage(content="""
Classify the customer intention.

Return ONLY one word from:

reservation
information
objection
comparison
complaint
agent
other
"""),

        HumanMessage(content=message)
    ]

    response = llm.invoke(messages)

    return response.content.strip().lower()


# -------------------------
# Chat principal
# -------------------------

@app.post("/chat")

async def chat(data: Message):

    if vector_db is None:

        return {
            "response": "Estoy cargando la información. Intenta nuevamente en unos segundos."
        }

    try:

        language = detect_language(data.message)

        intent = detect_intent(data.message)

        with index_lock:

            docs = vector_db.similarity_search(
                data.message,
                k=3
            )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        if data.user_id not in conversation_histories:

            conversation_histories[data.user_id] = []

        history = conversation_histories[data.user_id]

        history.append(
            HumanMessage(content=data.message)
        )

        if len(history) > MAX_HISTORY:

            history = history[-MAX_HISTORY:]

            conversation_histories[data.user_id] = history

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5
        )

        system_prompt = f"""
You are a professional hotel call center sales agent.

Customer language: {language}

Customer intent: {intent}

Use ONLY information from documents.

Context:
{context}

If the answer is not in the documents say:

"Esta información la consultare y le respondere en la brevedad."

Be professional, friendly and helpful.
"""

        messages = [

            SystemMessage(content=system_prompt),

            *history

        ]

        response = llm.invoke(messages)

        history.append(
            AIMessage(content=response.content)
        )

        return {

            "language_detected": language,

            "intent_detected": intent,

            "response": response.content

        }

    except Exception as e:

        logging.error(f"Error en chat: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# -------------------------
# Historial
# -------------------------

@app.get("/history/{user_id}")

async def get_history(user_id: str):

    history = conversation_histories.get(user_id, [])

    return [msg.content for msg in history]


# -------------------------
# Actualizar índice
# -------------------------

@app.post("/update-index")

async def update_index():

    threading.Thread(
        target=build_index,
        daemon=True
    ).start()

    return {
        "response": "Reconstrucción del índice iniciada"
    }


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
