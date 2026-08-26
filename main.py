import os
import uvicorn
import sqlite3
import logging
import threading
import json

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware  # <-- IMPORTANTE: Agregar esto
from dotenv import load_dotenv

# --- LIBRERÍAS OPENAI ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langdetect import detect, DetectorFactory

# ---------------- CONFIGURACIÓN ----------------

DetectorFactory.seed = 0
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI(title="Hotel AI Concierge")

# ---------------- CONFIGURACIÓN DE CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite peticiones desde Cloudflare Pages
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCUMENTS_DIR = "documents"
DATABASE = "hotel_memory.db"

# ---------------- ENDPOINT PARA EL FRONTEND WEB ----------------
@app.post("/api/get-vapi-token")
async def get_vapi_token():
    """Devuelve las llaves públicas de Vapi necesarias para la llamada en la web"""
    return {
        "publicKey": os.getenv("VAPI_PUBLIC_KEY", "3e179fa4-9b23-45a5-89fc-be35f7f7041f"),
        "assistantId": os.getenv("VAPI_ASSISTANT_ID", "ab2f3c03-6ebb-4d32-b867-9f355923f722")
    }
