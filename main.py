import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import threading
import time
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

# --- Configuración de logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Call Center IA Mejorado")

class Message(BaseModel):
    user_id: str
    message: str

INDEX_FILE = "hotel_faiss_index"
DOCUMENTS_DIR = os.path.join(os.getcwd(), "documents")

vector_db = None
index_lock = threading.Lock()
conversation_histories: Dict[str, List[Dict]] = {}  # Historial por user_id

def build_index():
    """
    Construye o carga el índice FAISS desde todos los PDFs en documents/
    """
    global vector_db
    try:
        embeddings = OpenAIEmbeddings()

        if os.path.exists(INDEX_FILE):
            logging.info(f"🔄 Cargando índice FAISS existente desde {INDEX_FILE}")
            vector_db = FAISS.load_local(INDEX_FILE, embeddings, allow_dangerous_deserialization=True)
        else:
            logging.info("📄 Creando índice FAISS desde PDFs en 'documents/'...")
            docs = []
            for filename in os.listdir(DOCUMENTS_DIR):
                if filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(DOCUMENTS_DIR, filename))
                    docs.extend(loader.load())
            if not docs:
                logging.warning("❌ No se encontraron PDFs en la carpeta documents/")
                vector_db = None
                return
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            vector_db = FAISS.from_documents(chunks, embeddings)
            vector_db.save_local(INDEX_FILE)
            logging.info(f"✅ Índice FAISS creado y guardado en {INDEX_FILE}")
    except Exception as e:
        logging.error(f"❌ Error al inicializar FAISS: {e}")
        vector_db = None

# --- Inicialización en un hilo para no bloquear el startup ---
@app.on_event("startup")
async def startup_event():
    threading.Thread(target=build_index, daemon=True).start()

@app.post("/chat")
async def chat(data: Message):
    if vector_db is None:
        return {"response": "Hola, estoy cargando la información. Por favor intenta en unos segundos."}

    try:
        # Búsqueda semántica
        with index_lock:
            docs = vector_db.similarity_search(data.message, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Construcción del historial
        if data.user_id not in conversation_histories:
            conversation_histories[data.user_id] = []
        conversation_histories[data.user_id].append({"role": "user", "content": data.message})

        # Llamada a LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        messages = [
            SystemMessage(content=(
                "Eres el asistente de Whala! Bávaro. Usa esta información para responder:\n"
                f"{context}\n"
                "Si no sabes la respuesta, ofrece pasar con un agente humano."
            )),
            HumanMessage(content=data.message)
        ]

        response = llm.invoke(messages)
        # Guardar respuesta en historial
        conversation_histories[data.user_id].append({"role": "assistant", "content": response.content})

        return {"response": response.content}

    except Exception as e:
        logging.error(f"❌ Error en /chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando la solicitud: {e}")

@app.post("/update-index")
async def update_index():
    """
    Endpoint para reconstruir el índice sin reiniciar el servidor
    """
    threading.Thread(target=build_index, daemon=True).start()
    return {"response": "Reconstrucción del índice iniciada en segundo plano."}

@app.get("/history/{user_id}")
async def get_history(user_id: str):
    """
    Retorna el historial de conversación de un usuario
    """
    return conversation_histories.get(user_id, [])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
