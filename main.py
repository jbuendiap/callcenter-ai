from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
import os

app = FastAPI()

class Message(BaseModel):
    message: str

# Variable global para el conocimiento
knowledge = ""

# Usar una ruta más segura para Railway
PDF_PATH = os.path.join(os.getcwd(), "documents", "hotel_info.pdf")

@app.on_event("startup")
async def load_knowledge():
    global knowledge
    try:
        if os.path.exists(PDF_PATH):
            loader = PyPDFLoader(PDF_PATH)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(documents)
            knowledge = " ".join([doc.page_content for doc in texts]).lower()
            print("✅ Conocimiento del hotel cargado con éxito")
        else:
            print(f"❌ Error: No se encontró el archivo en {PDF_PATH}")
    except Exception as e:
        print(f"❌ Error al cargar el PDF: {e}")

@app.post("/")
async def hotel_ai(data: Message):
    question = data.message.lower()

    # Lógica de respuestas rápidas
    if any(word in question for word in ["precio", "cuesta", "tarifa"]):
        return {"response": "Las habitaciones en Whala! comienzan desde 120 dólares por noche."}

    if any(word in question for word in ["reservar", "reserva", "booking"]):
        return {"response": "Claro, puedo ayudarte con tu reserva. ¿Para qué fecha deseas la habitación?"}

    if any(word in question for word in ["servicios", "piscina", "wifi"]):
        return {"response": "El hotel cuenta con piscina, wifi gratis, restaurante y transporte al aeropuerto."}

    # Búsqueda simple en el conocimiento del PDF
    if knowledge and question in knowledge:
        # Esto busca la frase en el texto y devuelve un pedazo relevante
        start_idx = knowledge.find(question)
        relevant_text = knowledge[start_idx:start_idx+400]
        return {"response": f"Encontré esto en la info del hotel: {relevant_text}..."}

    return {"response": "Con gusto te ayudo. ¿Podrías darme más detalles de tu consulta?"}
