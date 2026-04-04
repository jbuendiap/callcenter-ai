import os
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI()

class Message(BaseModel):
    message: str

vector_db = None


@app.on_event("startup")
async def load_pdf():

    global vector_db

    try:

        path = os.path.join(os.getcwd(), "documents", "hotel_info.pdf")

        loader = PyPDFLoader(path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()

        vector_db = FAISS.from_documents(chunks, embeddings)

        print("✅ PDF convertido a base de conocimiento IA")

    except Exception as e:

        print(f"❌ Error cargando PDF: {e}")


@app.post("/")
async def hotel_ai(data: Message):

    if vector_db is None:
        return {"response": "La base de datos aún se está cargando."}

    try:

        docs = vector_db.similarity_search(data.message, k=3)

        context = "\n\n".join([doc.page_content for doc in docs])

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5
        )

        messages = [

            SystemMessage(content=f"""
Eres el asistente virtual del hotel Whala! Bávaro.

Responde de forma natural, amable y clara.

Usa SOLO la información proporcionada.

Información del hotel:
{context}

Si la información no está disponible,
di que puedes transferir la llamada a recepción.
"""),

            HumanMessage(content=data.message)

        ]

        response = llm.invoke(messages)

        return {"response": response.content}

    except Exception as e:

        return {"response": f"Error: {e}"}
