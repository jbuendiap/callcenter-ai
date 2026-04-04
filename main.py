import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# --- IMPORTACIONES CORREGIDAS PARA LANGCHAIN MODERNO ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter # Cambio aquí
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI()

class Message(BaseModel):
    message: str

vector_db = None

@app.on_event("startup")
async def load_pdf():
    global vector_db
    try:
        # Ruta al PDF dentro de la carpeta documents
        path = os.path.join(os.getcwd(), "documents", "hotel_info.pdf")
        
        if not os.path.exists(path):
            print(f"❌ Error: No se encuentra el archivo en {path}")
            return

        # 1. Cargar PDF
        loader = PyPDFLoader(path)
        docs = loader.load()

        # 2. Dividir texto
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        # 3. Crear base de datos vectorial
        embeddings = OpenAIEmbeddings() # Usa la variable OPENAI_API_KEY de Railway
        vector_db = FAISS.from_documents(chunks, embeddings)

        print("✅ Base de conocimientos cargada correctamente.")

    except Exception as e:
        print(f"❌ Error en el inicio: {e}")

@app.post("/")
async def hotel_ai(data: Message):
    if vector_db is None:
        return {"response": "Hola, estoy cargando la información. Un momento por favor."}

    try:
        # 1. Buscar en el PDF
        docs = vector_db.similarity_search(data.message, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 2. IA de respuesta
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

        messages = [
            SystemMessage(content=f"""
                Eres el asistente de Whala! Bávaro. 
                Usa esta información para responder:
                {context}
                
                Si no sabes la respuesta, ofrece pasar con un agente humano.
            """),
            HumanMessage(content=data.message)
        ]

        response = llm.invoke(messages)
        return {"response": response.content}

    except Exception as e:
        return {"response": f"Lo siento, hubo un error: {e}"}

# Para que Railway asigne el puerto correctamente
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
