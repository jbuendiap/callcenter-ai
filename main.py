import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI()

# Configuración de la API Key (Debes ponerla en las variables de entorno de Railway)
os.environ["OPENAI_API_KEY"] = "TU_API_KEY_AQUI"

class Message(BaseModel):
    message: str

# Variable para guardar el texto del PDF
hotel_knowledge = ""

@app.on_event("startup")
async def load_pdf():
    global hotel_knowledge
    try:
        path = os.path.join(os.getcwd(), "documents", "hotel_info.pdf")
        loader = PyPDFLoader(path)
        docs = loader.load()
        # Unimos todo el contenido del PDF en una sola base de conocimientos
        hotel_knowledge = "\n".join([doc.page_content for doc in docs])
        print("✅ Información del hotel cargada.")
    except Exception as e:
        print(f"❌ Error cargando PDF: {e}")

@app.post("/")
async def hotel_ai(data: Message):
    # Si no hay PDF, usamos una respuesta genérica
    if not hotel_knowledge:
        return {"response": "Lo siento, mi base de datos está en mantenimiento."}

    # Llamamos a la IA (GPT-4o mini es rápido y económico)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Creamos el "Prompt": Las instrucciones para la IA
    messages = [
        SystemMessage(content=f"""
            Eres el asistente virtual inteligente de Whala! Bávaro. 
            Tu objetivo es ser amable, servicial y natural.
            
            Usa EXCLUSIVAMENTE la siguiente información del hotel para responder:
            {hotel_knowledge}
            
            Si el cliente pregunta algo que NO está en el texto, di que no tienes esa 
            información y ofrécele contactar con un humano. 
            No inventes datos. Usa un tono profesional pero cercano.
        """),
        HumanMessage(content=data.message)
    ]

    try:
        response = llm.invoke(messages)
        return {"response": response.content}
    except Exception as e:
        return {"response": f"Hubo un error al procesar tu consulta: {e}"}
