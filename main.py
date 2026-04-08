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

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="AI Call Center")

class Message(BaseModel):
    user_id: str
    message: str

INDEX_FILE = "faiss_index"
DOCUMENTS_DIR = os.path.join(os.getcwd(), "documents")

vector_db = None
index_lock = threading.Lock()

conversation_histories: Dict[str, List[Dict]] = {}

# Crear modelo una sola vez
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5
)

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

            logging.info("Creando índice desde documentos")

            docs = []

            if not os.path.exists(DOCUMENTS_DIR):
                os.makedirs(DOCUMENTS_DIR)

            for filename in os.listdir(DOCUMENTS_DIR):

                if filename.lower().endswith(".pdf"):

                    loader = PyPDFLoader(os.path.join(DOCUMENTS_DIR, filename))
                    docs.extend(loader.load())

            if not docs:

                logging.warning("No se encontraron documentos")

                vector_db = None
                return

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(docs)

            vector_db = FAISS.from_documents(chunks, embeddings)

            vector_db.save_local(INDEX_FILE)

            logging.info("Índice creado correctamente")

    except Exception as e:

        logging.error(f"Error creando índice: {e}")

        vector_db = None


@app.on_event("startup")
async def startup_event():

    threading.Thread(target=build_index, daemon=True).start()


@app.post("/chat")
async def chat(data: Message):

    if vector_db is None:

        return {"response": "La información aún se está cargando."}

    try:

        with index_lock:

            docs = vector_db.similarity_search(data.message, k=3)

        if not docs:

            return {"response": "Esta información la consultaré y le responderé en la brevedad."}

        context = "\n\n".join([doc.page_content for doc in docs])

        if data.user_id not in conversation_histories:

            conversation_histories[data.user_id] = []

        conversation_histories[data.user_id].append({
            "role": "user",
            "content": data.message
        })

        history = conversation_histories[data.user_id][-6:]

        chat_history = []

        for msg in history:

            if msg["role"] == "user":

                chat_history.append(HumanMessage(content=msg["content"]))

            elif msg["role"] == "assistant":

                chat_history.append(AIMessage(content=msg["content"]))

        messages = [

            SystemMessage(content=(
                "Responde utilizando únicamente la información proporcionada en los documentos."
                "Si la información no está disponible responde:"
                "'Esta información la consultaré y le responderé en la brevedad.'"
            )),

            *chat_history,

            HumanMessage(content=f"Información relevante:\n{context}\n\nPregunta:\n{data.message}")

        ]

        response = llm.invoke(messages)

        conversation_histories[data.user_id].append({
            "role": "assistant",
            "content": response.content
        })

        return {"response": response.content}

    except Exception as e:

        logging.error(f"Error en chat: {e}")

        raise HTTPException(status_code=500, detail="Error procesando la solicitud")


@app.post("/update-index")
async def update_index():

    threading.Thread(target=build_index, daemon=True).start()

    return {"response": "Reconstrucción del índice iniciada."}


@app.get("/history/{user_id}")
async def get_history(user_id: str):

    return conversation_histories.get(user_id, [])


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
