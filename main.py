from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Call Center AI funcionando"}
from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Call Center AI funcionando"}

@app.post("/call")
async def receive_call(request: Request):
    data = await request.json()
    
    print("Llamada recibida:", data)

    return {
        "response": "Hola, gracias por llamar al hotel. ¿En qué puedo ayudarte?"
    }
