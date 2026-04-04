from fastapi import FastAPI
from pydantic import BaseModel
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

app = FastAPI()

class Message(BaseModel):
    message: str

vector_db = None
llm = None

PDF_PATH = os.path.join(os.getcwd(), "documents", "hotel_info.pdf")


@app.on_event("startup")
async def load_knowledge():

    global vector_db
    global llm

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vector_db = FAISS.from_documents(texts, embeddings)

    llm = ChatOpenAI(
        temperature=0.2
    )

    print("✅ IA del hotel lista")


@app.post("/")
async def hotel_ai(data: Message):

    question = data.message

    docs = vector_db.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Eres un asistente de hotel.

Usa la siguiente información para responder al cliente:

{context}

Pregunta del cliente:
{question}
"""

    response = llm.invoke(prompt)

    return {"response": response.content}
