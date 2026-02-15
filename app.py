import os
import sys
import requests
import streamlit as st

# Fix sqlite issue
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage


# =========================================================
# CONFIG
# =========================================================

GITHUB_REPO = "https://github.com/rahulpakhare4/mydoc/tree/main"
PDF_FILES = [
    "Rahul%20Pakhare.pdf"    
]

# =========================================================
# LOAD API KEY
# =========================================================

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in secrets.toml")
    st.stop()


# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    chat = ChatGroq(
        temperature=0.3,
        model="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY
    )

    chroma_client = chromadb.Client(
        chromadb.Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        )
    )

    collection = chroma_client.get_or_create_collection("knowledge")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return embedding_model, chat, collection, memory


embedding_model, chat, collection, memory = load_models()


# =========================================================
# DOWNLOAD PDF FROM GITHUB
# =========================================================
def download_pdf(url):
    response = requests.get(url)
    return response.content


# =========================================================
# READ PDF
# =========================================================
def load_pdf_from_bytes(pdf_bytes):
    reader = PdfReader(pdf_bytes)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# =========================================================
# TEXT SPLIT
# =========================================================
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    return splitter.split_text(text)


# =========================================================
# LOAD ALL PDFS FROM REPO
# =========================================================
def load_repo_pdfs():

    if collection.count() > 0:
        return

    all_chunks = []

    for file in PDF_FILES:
        url = GITHUB_REPO + file

        try:
            pdf_bytes = download_pdf(url)
            text = load_pdf_from_bytes(pdf_bytes)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)

        except Exception as e:
            st.warning(f"Failed to load {file}: {e}")

    if not all_chunks:
        st.error("No PDFs loaded")
        return

    embeddings = embedding_model.embed_documents(all_chunks)

    collection.add(
        ids=[str(i) for i in range(len(all_chunks))],
        documents=all_chunks,
        embeddings=embeddings
    )


# Load PDFs automatically
load_repo_pdfs()


# =========================================================
# RETRIEVE CONTEXT
# =========================================================
def retrieve_context(query):

    if collection.count() == 0:
        return ["No documents available"]

    q_embed = embedding_model.embed_query(query)

    results = collection.query(
        query_embeddings=[q_embed],
        n_results=2
    )

    return results["documents"][0]


# =========================================================
# ASK AI
# =========================================================
def ask_ai(question):

    system_prompt = """
You are an AI clone of Rahul Pakhare, a GIS consultant.
Speak professionally and concisely.
Never hallucinate.
"""

    history = memory.load_memory_variables({}).get("chat_history", [])[-6:]
    context = retrieve_context(question)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"History:{history}\nContext:{context}\nQ:{question}")
    ]

    try:
        response = chat.invoke(messages)
        answer = response.content
    except Exception as e:
        return f"Model error: {e}"

    memory.save_context(
        {"input": question},
        {"output": answer}
    )

    return answer


# =========================================================
# UI
# =========================================================
st.title("🤖 Rahul AI Clone")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("Ask something...")

if question:
    st.session_state.history.append({"role": "user", "content": question})

    with st.spinner("Thinking..."):
        answer = ask_ai(question)

    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()
