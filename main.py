import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import threading
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AI Call Center")

class Message(BaseModel):
    user_id: str
    message: str


DOCUMENTS_DIR = "documents"

INFO_INDEX = "info_index"
QUESTIONS_INDEX = "questions_index"

info_db = None
questions_db = None

index_lock = threading.Lock()

conversation_histories: Dict[str, List[Dict]] = {}

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4
)


def load_pdfs(folder):

    docs = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(".pdf"):

            loader = PyPDFLoader(os.path.join(folder, filename))

            docs.extend(loader.load())

    return docs


def build_indexes():

    global info_db, questions_db

    embeddings = OpenAIEmbeddings()

    try:

        # índice de información

        info_docs = load_pdfs(os.path.join(DOCUMENTS_DIR, "info"))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        info_chunks = splitter.split_documents(info_docs)

        info_db = FAISS.from_documents(info_chunks, embeddings)

        info_db.save_local(INFO_INDEX)

        # índice de preguntas

        questions_docs = load_pdfs(os.path.join(DOCUMENTS_DIR, "questions"))

        questions_chunks = splitter.split_documents(questions_docs)

        questions_db = FAISS.from_documents(questions_chunks, embeddings)

        questions_db.save_local(QUESTIONS_INDEX)

        logging.info("Indexes created successfully")

    except Exception as e:

        logging.error(e)


@app.on_event("startup")
async def startup():

    threading.Thread(target=build_indexes, daemon=True).start()


@app.post("/chat")
async def chat(data: Message):

    if info_db is None:

        return {"response": "Loading information, please try again shortly."}

    with index_lock:

        info_docs = info_db.similarity_search(data.message, k=3)

        question_docs = questions_db.similarity_search(data.message, k=2)

    info_context = "\n\n".join([doc.page_content for doc in info_docs])

    question_context = "\n\n".join([doc.page_content for doc in question_docs])

    if data.user_id not in conversation_histories:

        conversation_histories[data.user_id] = []

    conversation_histories[data.user_id].append({
        "role": "user",
        "content": data.message
    })

    history = conversation_histories[data.user_id][-6:]

    chat_history = []

    for msg in history:

        if msg["role"] == "user":

            chat_history.append(HumanMessage(content=msg["content"]))

        else:

            chat_history.append(AIMessage(content=msg["content"]))

    messages = [

        SystemMessage(content="""
Detect the user's language and respond in the same language.

Use the information from the hotel documents to answer questions.

Use the question flow document to guide the conversation and ask relevant questions.

If the information is not available respond:
"Esta información la consultaré y le responderé en la brevedad."
"""),

        *chat_history,

        HumanMessage(content=f"""
Hotel information:
{info_context}

Question flow:
{question_context}

User message:
{data.message}
""")

    ]

    response = llm.invoke(messages)

    conversation_histories[data.user_id].append({
        "role": "assistant",
        "content": response.content
    })

    return {"response": response.content}


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(app, host="0.0.0.0", port=port)
