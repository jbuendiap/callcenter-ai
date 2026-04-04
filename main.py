import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Librerías de LangChain para RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI()

class Message(BaseModel):
    message: str

# Variable global para la base de datos de vectores
vector_db = None

@app.on_event("startup")
async def load_pdf():
    global vector_db
    try:
        # Ruta dinámica compatible con Railway
        path = os.path.join(os.getcwd(), "documents", "hotel_info.pdf")
        
        if not os.path.exists(path):
            print(f"❌ Error: El archivo no existe en la ruta: {path}")
            return

        # 1. Cargar el PDF
        loader = PyPDFLoader(path)
        docs = loader.load()

        # 2. Dividir el texto en fragmentos inteligentes
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        # 3. Crear Embeddings y Base de Datos Vectorial (FAISS)
        # Nota: Requiere OPENAI_API_KEY en las variables de entorno de Railway
        embeddings = OpenAIEmbeddings()
        vector_db = FAISS.from_documents(chunks, embeddings)

        print("✅ PDF cargado y convertido a base de conocimiento IA con éxito.")

    except Exception as e:
        print(f"❌ Error crítico en el inicio: {e}")

@app.post("/")
async def hotel_ai(data: Message):
    # Verificación de seguridad por si el PDF aún no carga
    if vector_db is None:
        return {"response": "Hola, estoy terminando de preparar la información del hotel. Por favor, intenta de nuevo en unos segundos."}

    try:
        # 1. Buscar los 3 fragmentos más relevantes en el PDF
        docs = vector_db.similarity_search(data.message, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 2. Configurar el modelo de lenguaje (GPT-4o-mini es ideal para Vapi por su velocidad)
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5
        )

        # 3. Crear el mensaje del sistema con el contexto del hotel
        messages = [
            SystemMessage(content=f"""
                Eres el asistente virtual del hotel Whala! Bávaro.
                Tu tono debe ser amable, servicial y profesional.
                
                Instrucciones:
                - Usa ÚNICAMENTE la información proporcionada abajo para responder.
                - Si la respuesta no está en el texto, di amablemente que no tienes el detalle exacto pero que puedes transferir la llamada a un agente humano.
                - Mantén las respuestas concisas para que la voz de Vapi sea fluida.

                Información relevante del hotel:
                {context}
            """),
            HumanMessage(content=data.message)
        ]

        # 4. Obtener respuesta de la IA
        response = llm.invoke(messages)

        # 5. Formato de retorno compatible con Vapi
        return {"response": response.content}

    except Exception as e:
        print(f"❌ Error procesando mensaje: {e}")
        return {"response": "Lo siento, tuve un pequeño inconveniente técnico. ¿Podrías repetirme tu pregunta?"}

# Bloque de ejecución para Railway
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
