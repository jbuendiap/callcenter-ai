import os
import uvicorn
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
from langdetect import detect

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# CONFIG
# =========================

os.environ["OPENAI_API_KEY"] = "TU_API_KEY"

app = FastAPI()

# =========================
# MODELO IA
# =========================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

# =========================
# BASE DE DATOS CRM
# =========================

conn = sqlite3.connect("crm.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS clientes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT,
    idioma TEXT,
    score INTEGER,
    emocion TEXT,
    listo_cerrar INTEGER
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS conversaciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cliente TEXT,
    mensaje TEXT,
    respuesta TEXT
)
""")

conn.commit()

# =========================
# CARGAR PDF DE VENTAS
# =========================

loader = PyPDFLoader("tecnicas_psicologicas_ventas_callcenter_ia.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()

# =========================
# MODELO DE DATOS
# =========================

class ClienteMensaje(BaseModel):
    nombre: str
    mensaje: str

# =========================
# DETECCION DE INTENCION
# =========================

def detectar_intencion(mensaje):

    prompt = f"""
    Clasifica la intención del cliente.

    Opciones:

    informacion
    precio
    duda
    objecion
    compra

    Mensaje:
    {mensaje}
    """

    result = llm.invoke(prompt)

    return result.content.strip().lower()

# =========================
# DETECCION EMOCIONAL
# =========================

def detectar_emocion(mensaje):

    prompt = f"""
    Detecta la emoción del cliente.

    Opciones:

    interesado
    dudoso
    confundido
    frustrado
    listo_comprar

    Mensaje:
    {mensaje}
    """

    result = llm.invoke(prompt)

    return result.content.strip().lower()

# =========================
# LEAD SCORING
# =========================

def calcular_score(intencion, emocion):

    score = 0

    if intencion == "precio":
        score += 30

    if intencion == "compra":
        score += 50

    if emocion == "interesado":
        score += 20

    if emocion == "listo_comprar":
        score += 40

    return score

# =========================
# DETECTAR SI CERRAR VENTA
# =========================

def predecir_cierre(score):

    if score >= 70:
        return True
    else:
        return False

# =========================
# MEMORIA DEL CLIENTE
# =========================

def guardar_conversacion(cliente, mensaje, respuesta):

    cursor.execute("""
    INSERT INTO conversaciones (cliente, mensaje, respuesta)
    VALUES (?, ?, ?)
    """, (cliente, mensaje, respuesta))

    conn.commit()

# =========================
# RESPUESTA IA
# =========================

def generar_respuesta(mensaje):

    docs = retriever.get_relevant_documents(mensaje)

    contexto = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Usa la siguiente información para responder al cliente.

    {contexto}

    Cliente pregunta:
    {mensaje}
    """

    result = llm.invoke(prompt)

    return result.content

# =========================
# ENDPOINT PRINCIPAL
# =========================

@app.post("/chat")
def chat(data: ClienteMensaje):

    idioma = detect(data.mensaje)

    intencion = detectar_intencion(data.mensaje)

    emocion = detectar_emocion(data.mensaje)

    score = calcular_score(intencion, emocion)

    listo_cerrar = predecir_cierre(score)

    respuesta = generar_respuesta(data.mensaje)

    if listo_cerrar:

        respuesta += """

        Podemos activar el servicio ahora mismo.
        ¿Deseas que te ayude a completar la compra?
        """

    cursor.execute("""
    INSERT INTO clientes (nombre, idioma, score, emocion, listo_cerrar)
    VALUES (?, ?, ?, ?, ?)
    """, (data.nombre, idioma, score, emocion, int(listo_cerrar)))

    conn.commit()

    guardar_conversacion(data.nombre, data.mensaje, respuesta)

    return {
        "respuesta": respuesta,
        "idioma": idioma,
        "intencion": intencion,
        "emocion": emocion,
        "lead_score": score,
        "listo_para_cerrar": listo_cerrar
    }

# =========================
# INICIAR SERVIDOR
# =========================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
