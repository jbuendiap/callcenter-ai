import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI()

class Message(BaseModel):
    message: str

vector_db = None
INDEX_FILE = "hotel_faiss_index"
PDF_FILE = os.path.join(os.getcwd(), "documents", "hotel_info.pdf")

@app.on_event("startup")
async def startup_event():
    global vector_db
    try:
        embeddings = OpenAIEmbeddings()

        if os.path.exists(INDEX_FILE):
            # Se requiere allow_dangerous_deserialization para cargar índices locales
            vector_db = FAISS.load_local(INDEX_FILE, embeddings, allow_dangerous_deserialization=True)
            print(f"✅ Índice FAISS cargado desde {INDEX_FILE}")
        elif os.path.exists(PDF_FILE):
            loader = PyPDFLoader(PDF_FILE)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(docs)

            vector_db = FAISS.from_documents(chunks, embeddings)
            vector_db.save_local(INDEX_FILE)
            print(f"✅ Índice FAISS creado y guardado en {INDEX_FILE}")
        else:
            print(f"❌ No se encuentra el PDF en {PDF_FILE}.")
            vector_db = None

    except Exception as e:
        print(f"❌ Error al inicializar FAISS: {e}")
        vector_db = None

@app.post("/")
async def hotel_ai(data: Message):
    if vector_db is None:
        return {"response": "Hola, estoy cargando la información. Por favor intenta en unos segundos."}

    try:
        docs = vector_db.similarity_search(data.message, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        messages = [
            SystemMessage(content=(
                "Eres el asistente de Whala! Bávaro. Usa esta información para responder:\n"
                f"{context}\n"
                "Si no sabes la respuesta, ofrece pasar con un agente humano."
            )),
            HumanMessage(content=data.message)
        ]

        # Uso de .invoke() para compatibilidad
        response = llm.invoke(messages)
        return {"response": response.content}

    except Exception as e:
        return {"response": f"Lo siento, hubo un error: {e}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
