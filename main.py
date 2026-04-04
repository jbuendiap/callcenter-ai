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

        # dividir el texto en pedazos
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        documents = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()

        vector_db = FAISS.from_documents(documents, embeddings)

        print("✅ Base de conocimiento creada.")

    except Exception as e:
        print(f"❌ Error cargando PDF: {e}")


@app.post("/")
async def hotel_ai(data: Message):

    global vector_db

    if not vector_db:
        return {"response": "La base de datos aún no está lista."}

    try:

        docs = vector_db.similarity_search(data.message, k=3)

        context = "\n".join([doc.page_content for doc in docs])

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5
        )

        messages = [

            SystemMessage(content=f"""
Eres el asistente virtual del hotel Whala! Bávaro.

Responde de forma natural, amable y útil.

Usa SOLO la información del hotel que aparece abajo.

Información del hotel:
{context}

Si el usuario pregunta algo que no está en el contexto, dile que no tienes esa información y ofrece ayuda adicional.
"""),

            HumanMessage(content=data.message)

        ]

        response = llm.invoke(messages)

        return {"response": response.content}

    except Exception as e:
        return {"response": str(e)}
