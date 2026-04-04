from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

app = FastAPI()

class Message(BaseModel):
    message: str


# cargar el PDF
loader = PyPDFLoader("documents/hotel_info.pdf")
documents = loader.load()

# dividir el texto
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = text_splitter.split_documents(documents)

# guardar conocimiento
knowledge = " ".join([doc.page_content for doc in texts]).lower()


@app.post("/")
async def hotel_ai(data: Message):

    question = data.message.lower()

    # ejemplos de respuestas inteligentes
    if "precio" in question or "cuesta" in question:
        return {"response": "Las habitaciones comienzan desde 120 dólares por noche."}

    if "reservar" in question or "reserva" in question:
        return {"response": "Claro, puedo ayudarte con tu reserva. ¿Para qué fecha deseas la habitación?"}

    if "servicios" in question or "piscina" in question:
        return {"response": "El hotel cuenta con piscina, wifi gratis, restaurante y transporte al aeropuerto."}

    if question in knowledge:
        return {"response": "Según la información del hotel: " + knowledge[:300]}

    return {"response": "Con gusto te ayudo. ¿Podrías darme más detalles de tu consulta?"}
