import os
import uvicorn
import sqlite3
import logging
import threading
import json
import requests

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage
)

from langdetect import detect, DetectorFactory


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

DetectorFactory.seed = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ==========================================================
# APLICACIÓN
# ==========================================================

app = FastAPI(
    title="Asistente Virtual",
    version="2.0.0"
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# VARIABLES DE RAILWAY
# ==========================================================

BOT_ACTIVE = os.getenv(
    "BOT_ACTIVE",
    "true"
)

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

PORT = os.getenv(
    "PORT",
    "8000"
)

TELEGRAM_TOKEN = os.getenv(
    "TELEGRAM_TOKEN"
)

# IMPORTANTE:
# PRIVATE KEY SOLO SE UTILIZA EN EL BACKEND.
VAPI_PRIVATE_KEY = os.getenv(
    "VAPI_PRIVATE_KEY"
)

# PUBLIC KEY:
# Se utiliza para inicializar Vapi Web SDK en el navegador.
# Sigue siendo gestionada desde Railway.
VAPI_PUBLIC_KEY = os.getenv(
    "VAPI_PUBLIC_KEY"
)


# ==========================================================
# ASSISTANTS VAPI
# ==========================================================

VAPI_ASSISTANTS = {

    "es": os.getenv(
        "VAPI_ASSISTANT_ES"
    ),

    "en": os.getenv(
        "VAPI_ASSISTANT_EN"
    ),

    "fr": os.getenv(
        "VAPI_ASSISTANT_FR"
    ),

    "ru": os.getenv(
        "VAPI_ASSISTANT_RU"
    )
}


# ==========================================================
# RUTAS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DOCUMENTS_DIR = os.path.join(
    BASE_DIR,
    "documents"
)

DATABASE = os.path.join(
    BASE_DIR,
    "hotel_memory.db"
)


# ==========================================================
# VARIABLES GLOBALES
# ==========================================================

vector_db = None

index_lock = threading.Lock()

active_vapi_call_id = None

active_vapi_language = None

vapi_call_lock = threading.Lock()


# ==========================================================
# BOT ACTIVE
# ==========================================================

def is_bot_active():

    value = str(
        BOT_ACTIVE
    ).strip().lower()

    return value in (
        "true",
        "1",
        "yes",
        "on",
        "active"
    )


# ==========================================================
# NORMALIZAR IDIOMA
# ==========================================================

def normalize_language(language):

    if not language:

        return "es"

    language = str(
        language
    ).lower().strip()

    if language.startswith("es"):

        return "es"

    if language.startswith("en"):

        return "en"

    if language.startswith("fr"):

        return "fr"

    if language.startswith("ru"):

        return "ru"

    return "es"


# ==========================================================
# OBTENER ASSISTANT SEGÚN IDIOMA
# ==========================================================

def get_vapi_assistant(language):

    language = normalize_language(
        language
    )

    assistant_id = VAPI_ASSISTANTS.get(
        language
    )

    if not assistant_id:

        logging.warning(
            "No existe VAPI_ASSISTANT_%s. "
            "Intentando utilizar español.",
            language.upper()
        )

        language = "es"

        assistant_id = VAPI_ASSISTANTS.get(
            "es"
        )

    return language, assistant_id


# ==========================================================
# DETECTAR IDIOMA DEL NAVEGADOR
# ==========================================================

def detect_browser_language(
    request: Request
):

    header = request.headers.get(
        "accept-language",
        ""
    )

    if not header:

        return "es"

    first_language = (
        header
        .split(",")[0]
        .split("-")[0]
        .strip()
        .lower()
    )

    return normalize_language(
        first_language
    )


# ==========================================================
# ROOT
# ==========================================================

@app.get("/")
def home():

    return {
        "status": "ok",
        "service": "Asistente Virtual",
        "backend": "Railway",
        "message": "Backend funcionando correctamente."
    }


# ==========================================================
# BASE DE DATOS
# ==========================================================

def init_db():

    try:

        conn = sqlite3.connect(
            DATABASE,
            check_same_thread=False
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.commit()

        conn.close()

        logging.info(
            "Base de datos SQLite inicializada correctamente."
        )

    except Exception as e:

        logging.error(
            "Error inicializando SQLite: %s",
            str(e)
        )


# ==========================================================
# GUARDAR MENSAJE
# ==========================================================

def save_message(
    user_id: str,
    role: str,
    message: str
):

    try:

        conn = sqlite3.connect(
            DATABASE,
            check_same_thread=False
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO conversations
            (user_id, role, message)
            VALUES (?, ?, ?)
            """,
            (
                user_id,
                role,
                message
            )
        )

        conn.commit()

        conn.close()

    except Exception as e:

        logging.error(
            "Error guardando mensaje: %s",
            str(e)
        )


# ==========================================================
# HISTORIAL
# ==========================================================

def get_history(
    user_id: str,
    limit: int = 6
):

    try:

        conn = sqlite3.connect(
            DATABASE,
            check_same_thread=False
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT role, message
            FROM conversations
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                user_id,
                limit
            )
        )

        rows = cursor.fetchall()

        conn.close()

        rows.reverse()

        history = []

        for role, message in rows:

            if role == "user":

                history.append(
                    HumanMessage(
                        content=message
                    )
                )

            elif role == "assistant":

                history.append(
                    AIMessage(
                        content=message
                    )
                )

        return history

    except Exception as e:

        logging.error(
            "Error obteniendo historial: %s",
            str(e)
        )

        return []


# ==========================================================
# DETECTAR IDIOMA DEL MENSAJE
# ==========================================================

def detect_language(
    message: str
):

    try:

        if not message:

            return "es"

        if len(
            message.strip()
        ) < 3:

            return "es"

        language = detect(
            message
        )

        return normalize_language(
            language
        )

    except Exception as e:

        logging.warning(
            "No se pudo detectar idioma: %s",
            str(e)
        )

        return "es"


# ==========================================================
# BUSCAR EN DOCUMENTOS
# ==========================================================

def search_documents(
    query: str,
    number_of_documents: int = 4
):

    global vector_db

    if vector_db is None:

        logging.warning(
            "El índice FAISS todavía no está disponible."
        )

        return ""

    try:

        with index_lock:

            documents = vector_db.similarity_search(
                query,
                k=number_of_documents
            )

        if not documents:

            return ""

        context_parts = []

        for document in documents:

            content = (
                document.page_content.strip()
            )

            if not content:

                continue

            source = document.metadata.get(
                "source",
                ""
            )

            document_name = document.metadata.get(
                "document_name",
                ""
            )

            page = document.metadata.get(
                "page",
                None
            )

            if document_name:

                source_name = document_name

            elif source:

                source_name = os.path.basename(
                    source
                )

            else:

                source_name = "Documento"

            if page is not None:

                try:

                    page_number = (
                        int(page) + 1
                    )

                except Exception:

                    page_number = page

                header = (
                    f"[Documento: {source_name} | "
                    f"Página: {page_number}]"
                )

            else:

                header = (
                    f"[Documento: {source_name}]"
                )

            context_parts.append(
                f"{header}\n{content}"
            )

        return "\n\n".join(
            context_parts
        )

    except Exception as e:

        logging.error(
            "Error buscando documentos: %s",
            str(e)
        )

        return ""


# ==========================================================
# PROCESAR MENSAJE
# ==========================================================

def process_message(
    user_id: str,
    message: str
):

    try:

        if not message or not message.strip():

            return (
                "No recibí ninguna pregunta."
            )

        message = message.strip()

        language = detect_language(
            message
        )

        logging.info(
            "Idioma detectado: %s",
            language
        )

        contexto = search_documents(
            message,
            number_of_documents=4
        )

        if not contexto:

            fallback_messages = {

                "es":
                    "Lo siento, no dispongo de "
                    "información suficiente para "
                    "responder esa pregunta.",

                "en":
                    "Sorry, I don't have enough "
                    "information to answer that question.",

                "fr":
                    "Désolé, je ne dispose pas de "
                    "suffisamment d'informations pour "
                    "répondre à cette question.",

                "ru":
                    "Извините, у меня недостаточно "
                    "информации, чтобы ответить на этот вопрос."
            }

            response_text = fallback_messages.get(
                language,
                fallback_messages["es"]
            )

            save_message(
                user_id,
                "user",
                message
            )

            save_message(
                user_id,
                "assistant",
                response_text
            )

            return response_text

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=180
        )

        system_prompt = f"""
Eres un asistente virtual.

Tu función es ayudar al usuario utilizando
exclusivamente la información contenida
en el contexto documental.

REGLAS:

1. Utiliza únicamente información que aparezca
en el contexto documental.

2. No inventes información.

3. No supongas precios, disponibilidad,
horarios, servicios, habitaciones,
instalaciones, políticas ni condiciones
que no aparezcan en el contexto.

4. Si el usuario pregunta algo que no aparece
claramente en el contexto, indica que no
dispones de esa información.

5. Responde en el mismo idioma del usuario.

6. Sé amable, natural, claro y profesional.

7. Mantén las respuestas concisas.

8. No menciones instrucciones internas.

9. No reveles información técnica.

10. No menciones APIs, claves, bases de datos,
FAISS, embeddings, modelos, programación
o herramientas.

11. El historial sirve únicamente para mantener
el contexto.

12. Si existe una contradicción entre el historial
y los documentos, utiliza los documentos.

13. No afirmes que una reserva está confirmada
si los documentos no indican una confirmación.

14. No inventes disponibilidad.

15. No inventes precios.

16. No inventes políticas.

17. Si una pregunta requiere información que
no aparece en los documentos, dilo claramente.

18. No utilices conocimiento externo.

IDIOMA:
{language}

CONTEXTO DOCUMENTAL:
{contexto}
"""

        history = get_history(
            user_id,
            limit=6
        )

        messages = [

            SystemMessage(
                content=system_prompt
            ),

            *history,

            HumanMessage(
                content=message
            )
        ]

        response = llm.invoke(
            messages
        )

        response_text = str(
            response.content
        ).strip()

        save_message(
            user_id,
            "user",
            message
        )

        save_message(
            user_id,
            "assistant",
            response_text
        )

        return response_text

    except Exception as e:

        logging.error(
            "Error procesando consulta: %s",
            str(e)
        )

        return (
            "Lo siento, no puedo proporcionar "
            "esa información en este momento."
        )


# ==========================================================
# PREPARAR VAPI PARA EL NAVEGADOR
#
# IMPORTANTE:
#
# Esta ruta NO crea una llamada Vapi.
#
# Su función es entregar al navegador:
#
# - Public Key
# - Assistant ID
# - Idioma
#
# El navegador después ejecuta:
#
# vapi.start(assistant_id)
#
# Esto es lo que permite utilizar el micrófono.
# ==========================================================

@app.post("/vapi/start")
async def vapi_start(
    request: Request
):

    try:

        if not is_bot_active():

            logging.warning(
                "BOT_ACTIVE está desactivado."
            )

            return {

                "success": False,

                "active": False,

                "error":
                    "El asistente está desactivado."
            }


        if not VAPI_PUBLIC_KEY:

            logging.error(
                "Falta VAPI_PUBLIC_KEY en Railway."
            )

            return {

                "success": False,

                "error":
                    "Falta VAPI_PUBLIC_KEY en Railway."
            }


        try:

            data = await request.json()

        except Exception:

            data = {}


        if not isinstance(
            data,
            dict
        ):

            data = {}


        requested_language = (
            data.get("language")
        )


        if not requested_language:

            requested_language = (
                detect_browser_language(
                    request
                )
            )


        language, assistant_id = (
            get_vapi_assistant(
                requested_language
            )
        )


        if not assistant_id:

            logging.error(
                "No existe Assistant ID para idioma: %s",
                language
            )

            return {

                "success": False,

                "error":
                    "No existe un asistente Vapi configurado.",
            }


        logging.info(
            "Preparando Vapi Web."
        )

        logging.info(
            "Idioma seleccionado: %s",
            language
        )

        logging.info(
            "Assistant configurado para %s.",
            language
        )


        # NO SE ENVÍA PRIVATE KEY.
        return {

            "success":
                True,

            "active":
                True,

            "mode":
                "browser",

            "public_key":
                VAPI_PUBLIC_KEY,

            "assistant_id":
                assistant_id,

            "language":
                language
        }


    except Exception as e:

        logging.error(
            "Error preparando Vapi: %s",
            str(e)
        )

        return {

            "success":
                False,

            "error":
                "Error interno preparando el asistente."
        }


# ==========================================================
# REGISTRAR LLAMADA DEL NAVEGADOR
# ==========================================================

@app.post("/vapi/client-started")
async def vapi_client_started(
    request: Request
):

    global active_vapi_call_id
    global active_vapi_language

    try:

        data = await request.json()

        if not isinstance(
            data,
            dict
        ):

            data = {}


        call_id = data.get(
            "call_id"
        )

        language = normalize_language(
            data.get(
                "language",
                "es"
            )
        )


        with vapi_call_lock:

            active_vapi_call_id = (
                str(call_id)
                if call_id
                else None
            )

            active_vapi_language = language


        logging.info(
            "Llamada Vapi del navegador iniciada."
        )

        logging.info(
            "Idioma: %s",
            language
        )

        if call_id:

            logging.info(
                "Call ID: %s",
                call_id
            )


        return {

            "success":
                True
        }


    except Exception as e:

        logging.error(
            "Error registrando llamada cliente: %s",
            str(e)
        )

        return {

            "success":
                False
        }


# ==========================================================
# REGISTRAR FIN DE LLAMADA DEL NAVEGADOR
# ==========================================================

@app.post("/vapi/client-ended")
async def vapi_client_ended():

    global active_vapi_call_id
    global active_vapi_language

    with vapi_call_lock:

        active_vapi_call_id = None

        active_vapi_language = None


    logging.info(
        "Llamada Vapi del navegador finalizada."
    )


    return {

        "success":
            True
    }


# ==========================================================
# DETENER VAPI
#
# Esta ruta queda como respaldo.
#
# El navegador realmente debe ejecutar:
#
# vapi.stop()
#
# ==========================================================

@app.post("/vapi/stop")
async def vapi_stop():

    global active_vapi_call_id
    global active_vapi_language

    with vapi_call_lock:

        active_vapi_call_id = None

        active_vapi_language = None


    logging.info(
        "Estado Vapi limpiado."
    )


    return {

        "success":
            True,

        "message":
            "Estado de Vapi limpiado."
    }


# ==========================================================
# WEBHOOK VAPI
# ==========================================================

@app.post("/vapi-webhook")
async def vapi_webhook(
    request: Request
):

    try:

        data = await request.json()

        logging.info(
            "Webhook recibido de Vapi."
        )


        message_data = data.get(
            "message",
            {}
        )


        if not isinstance(
            message_data,
            dict
        ):

            return {
                "ok": True
            }


        message_type = (
            message_data.get(
                "type"
            )
        )


        logging.info(
            "Tipo de mensaje Vapi: %s",
            message_type
        )


        tool_calls = (
            message_data.get(
                "toolCalls",
                []
            )
        )


        if not tool_calls:

            return {
                "ok": True
            }


        tool_call = (
            tool_calls[0]
        )


        if not isinstance(
            tool_call,
            dict
        ):

            return {
                "ok": True
            }


        function_data = (
            tool_call.get(
                "function",
                {}
            )
        )


        if not isinstance(
            function_data,
            dict
        ):

            function_data = {}


        arguments = (
            function_data.get(
                "arguments",
                {}
            )
        )


        if isinstance(
            arguments,
            str
        ):

            try:

                arguments = json.loads(
                    arguments
                )

            except Exception:

                arguments = {
                    "query":
                        arguments
                }


        if not isinstance(
            arguments,
            dict
        ):

            arguments = {}


        query = arguments.get(
            "query",
            ""
        )


        if not isinstance(
            query,
            str
        ):

            query = str(
                query
            )


        query = query.strip()


        if not query:

            return {

                "results": [

                    {

                        "toolCallId":
                            tool_call.get("id"),

                        "result":
                            "No recibí ninguna pregunta."
                    }
                ]
            }


        customer_info = (
            message_data.get(
                "customer",
                {}
            )
        )


        if not isinstance(
            customer_info,
            dict
        ):

            customer_info = {}


        user_id = (

            customer_info.get(
                "number"
            )

            or

            customer_info.get(
                "id"
            )

            or

            "web_user"
        )


        user_id = str(
            user_id
        )


        respuesta = process_message(
            user_id,
            query
        )


        return {

            "results": [

                {

                    "toolCallId":
                        tool_call.get("id"),

                    "result":
                        respuesta
                }
            ]
        }


    except Exception as e:

        logging.error(
            "Error procesando webhook Vapi: %s",
            str(e)
        )

        return {

            "error":
                "Error procesando la solicitud."
        }


# ==========================================================
# ESTADO DE DOCUMENTOS
# ==========================================================

@app.get("/status-documents")
def status_documents():

    global vector_db

    try:

        if not os.path.exists(
            DOCUMENTS_DIR
        ):

            return {

                "status":
                    "error",

                "documents_folder":
                    DOCUMENTS_DIR,

                "folder_exists":
                    False,

                "pdfs":
                    [],

                "total_pdfs":
                    0,

                "index_ready":
                    vector_db is not None
            }


        files = os.listdir(
            DOCUMENTS_DIR
        )


        pdf_files = [

            file

            for file in sorted(
                files
            )

            if file.lower().endswith(
                ".pdf"
            )
        ]


        return {

            "status":
                "ok",

            "documents_folder":
                DOCUMENTS_DIR,

            "folder_exists":
                True,

            "pdfs":
                pdf_files,

            "total_pdfs":
                len(pdf_files),

            "index_ready":
                vector_db is not None
        }


    except Exception as e:

        return {

            "status":
                "error",

            "error":
                str(e)
        }


# ==========================================================
# HEALTH
# ==========================================================

@app.get("/health")
def health_check():

    global vector_db

    with vapi_call_lock:

        active_call = (
            active_vapi_call_id
        )

        active_language = (
            active_vapi_language
        )


    return {

        "status":
            "ok",

        "backend":
            "Railway",

        "bot_active":
            is_bot_active(),

        "documents_folder":
            DOCUMENTS_DIR,

        "index_ready":
            vector_db is not None,

        "openai_configured":
            bool(
                OPENAI_API_KEY
            ),

        "telegram_configured":
            bool(
                TELEGRAM_TOKEN
            ),

        "vapi_private_configured":
            bool(
                VAPI_PRIVATE_KEY
            ),

        "vapi_public_configured":
            bool(
                VAPI_PUBLIC_KEY
            ),

        "vapi_assistants": {

            "es":
                bool(
                    VAPI_ASSISTANTS.get("es")
                ),

            "en":
                bool(
                    VAPI_ASSISTANTS.get("en")
                ),

            "fr":
                bool(
                    VAPI_ASSISTANTS.get("fr")
                ),

            "ru":
                bool(
                    VAPI_ASSISTANTS.get("ru")
                )
        },

        "active_vapi_call":
            bool(
                active_call
            ),

        "active_vapi_language":
            active_language
    }


# ==========================================================
# CARGAR PDFS Y CREAR FAISS
# ==========================================================

def build_index():

    global vector_db

    try:

        logging.info(
            "=================================================="
        )

        logging.info(
            "INICIANDO CARGA DE DOCUMENTOS"
        )

        logging.info(
            "=================================================="
        )


        if not OPENAI_API_KEY:

            logging.error(
                "OPENAI_API_KEY no está configurada."
            )

            return


        logging.info(
            "OPENAI_API_KEY detectada."
        )


        logging.info(
            "Directorio principal: %s",
            BASE_DIR
        )


        logging.info(
            "Carpeta de documentos: %s",
            DOCUMENTS_DIR
        )


        if not os.path.exists(
            DOCUMENTS_DIR
        ):

            os.makedirs(
                DOCUMENTS_DIR
            )

            logging.warning(
                "La carpeta documents no existía."
            )

            logging.warning(
                "Fue creada automáticamente."
            )

            return


        files = os.listdir(
            DOCUMENTS_DIR
        )


        logging.info(
            "Archivos encontrados: %s",
            len(files)
        )


        if not files:

            logging.error(
                "La carpeta documents está VACÍA."
            )

            return


        for file in sorted(
            files
        ):

            full_path = os.path.join(
                DOCUMENTS_DIR,
                file
            )

            if os.path.isfile(
                full_path
            ):

                logging.info(
                    "Archivo encontrado: %s",
                    file
                )


        pdf_files = [

            file

            for file in sorted(
                files
            )

            if file.lower().endswith(
                ".pdf"
            )
        ]


        logging.info(
            "Archivos PDF encontrados: %s",
            len(pdf_files)
        )


        if not pdf_files:

            logging.error(
                "NO SE ENCONTRARON ARCHIVOS PDF."
            )

            return


        logging.info(
            "LISTA DE DOCUMENTOS PDF:"
        )


        for index, file in enumerate(
            pdf_files,
            start=1
        ):

            logging.info(
                "PDF %s: %s",
                index,
                file
            )


        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )


        all_docs = []

        total_pages = 0

        successful_pdfs = 0

        failed_pdfs = 0


        for file in pdf_files:

            file_path = os.path.join(
                DOCUMENTS_DIR,
                file
            )


            logging.info(
                "--------------------------------------------------"
            )


            logging.info(
                "LEYENDO PDF: %s",
                file
            )


            try:

                loader = PyPDFLoader(
                    file_path
                )

                documents = loader.load()

                pages = len(
                    documents
                )


                if pages == 0:

                    failed_pdfs += 1

                    continue


                for document in documents:

                    document.metadata[
                        "document_name"
                    ] = file

                    document.metadata[
                        "source_file"
                    ] = file


                total_pages += pages

                all_docs.extend(
                    documents
                )

                successful_pdfs += 1


                logging.info(
                    "PDF CARGADO CORRECTAMENTE: %s",
                    file
                )

                logging.info(
                    "Páginas: %s",
                    pages
                )


            except Exception as e:

                failed_pdfs += 1

                logging.error(
                    "ERROR LEYENDO PDF %s: %s",
                    file,
                    str(e)
                )


        logging.info(
            "=================================================="
        )

        logging.info(
            "PDF encontrados: %s",
            len(pdf_files)
        )

        logging.info(
            "PDF cargados: %s",
            successful_pdfs
        )

        logging.info(
            "PDF con errores: %s",
            failed_pdfs
        )

        logging.info(
            "Páginas cargadas: %s",
            total_pages
        )

        logging.info(
            "=================================================="
        )


        if not all_docs:

            logging.error(
                "No se pudo extraer contenido de los PDF."
            )

            return


        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100
        )


        chunks = splitter.split_documents(
            all_docs
        )


        logging.info(
            "Chunks creados: %s",
            len(chunks)
        )


        if not chunks:

            logging.error(
                "No se generaron chunks."
            )

            return


        logging.info(
            "Creando índice FAISS..."
        )


        new_vector_db = FAISS.from_documents(
            chunks,
            embeddings
        )


        with index_lock:

            vector_db = new_vector_db


        logging.info(
            "=================================================="
        )

        logging.info(
            "ÍNDICE FAISS CREADO CORRECTAMENTE"
        )

        logging.info(
            "PDF procesados: %s",
            successful_pdfs
        )

        logging.info(
            "Páginas procesadas: %s",
            total_pages
        )

        logging.info(
            "Chunks: %s",
            len(chunks)
        )

        logging.info(
            "RAG LISTO"
        )

        logging.info(
            "=================================================="
        )


    except Exception as e:

        logging.error(
            "ERROR CONSTRUYENDO ÍNDICE FAISS: %s",
            str(e)
        )


# ==========================================================
# STARTUP
# ==========================================================

@app.on_event(
    "startup"
)
async def startup():

    logging.info(
        "=================================================="
    )

    logging.info(
        "SERVICIO INICIANDO"
    )

    logging.info(
        "BACKEND: RAILWAY"
    )

    logging.info(
        "=================================================="
    )


    init_db()


    logging.info(
        "BOT_ACTIVE configurado: %s",
        is_bot_active()
    )


    if OPENAI_API_KEY:

        logging.info(
            "OPENAI_API_KEY detectada."
        )

    else:

        logging.error(
            "FALTA OPENAI_API_KEY."
        )


    if VAPI_PRIVATE_KEY:

        logging.info(
            "VAPI_PRIVATE_KEY detectada."
        )

    else:

        logging.error(
            "FALTA VAPI_PRIVATE_KEY."
        )


    if VAPI_PUBLIC_KEY:

        logging.info(
            "VAPI_PUBLIC_KEY detectada."
        )

    else:

        logging.error(
            "FALTA VAPI_PUBLIC_KEY."
        )


    if TELEGRAM_TOKEN:

        logging.info(
            "TELEGRAM_TOKEN detectado."
        )

    else:

        logging.warning(
            "TELEGRAM_TOKEN no configurado."
        )


    for language, assistant_id in (
        VAPI_ASSISTANTS.items()
    ):

        if assistant_id:

            logging.info(
                "VAPI_ASSISTANT_%s configurado.",
                language.upper()
            )

        else:

            logging.warning(
                "VAPI_ASSISTANT_%s no configurado.",
                language.upper()
            )


    logging.info(
        "BASE_DIR: %s",
        BASE_DIR
    )

    logging.info(
        "DOCUMENTS_DIR: %s",
        DOCUMENTS_DIR
    )

    logging.info(
        "DATABASE: %s",
        DATABASE
    )


    threading.Thread(
        target=build_index,
        daemon=True
    ).start()


# ==========================================================
# EJECUCIÓN
# ==========================================================

if __name__ == "__main__":

    port = int(
        PORT
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
