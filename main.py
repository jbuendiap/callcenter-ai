from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Message(BaseModel):
    message: str

@app.post("/")
async def hotel_ai(data: Message):
    text = data.message.lower()

    if "precio" in text or "cuesta" in text:
        return {"response": "Las habitaciones empiezan desde 120 dólares por noche."}

    if "reservar" in text or "reserva" in text:
        return {"response": "Claro, puedo ayudarte con tu reserva. ¿Para qué fecha deseas la habitación?"}

    if "servicios" in text:
        return {"response": "El hotel tiene piscina, wifi gratis, restaurante y transporte al aeropuerto."}

    return {"response": "Claro, con gusto te ayudo. ¿En qué puedo asistirte?"}
