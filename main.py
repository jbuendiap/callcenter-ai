from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

app = FastAPI()

class Message(BaseModel):
    message: str

# cargar el PDF del hotel
loader = PyPDFLoader("documents/hotel_info.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = text_splitter.split_documents(documents)

# unir el texto del pdf
knowledge = " ".join([doc.page_content for doc in texts])


@app.post("/")
async def hotel_ai(data: Message):

    question = data.message.lower()

    if question in knowledge.lower():
        return {"response": "Según la información del hotel: " + knowledge[:400]}

    return {"response": "Déjame verificar esa información para ayudarte mejor."}
