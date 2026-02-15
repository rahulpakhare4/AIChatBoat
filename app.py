import os
import sys
import streamlit as st

# Fix sqlite issue for Streamlit Cloud
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
import numpy as np
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage


# =========================================================
# CONFIG
# =========================================================

PDF_FILE = "Rahul Pakhare.pdf"


# =========================================================
# LOAD API KEY
# =========================================================

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in secrets.toml")
    st.stop()


# =========================================================
# SIMPLE LIGHTWEIGHT EMBEDDING
# (No heavy ML dependencies)
# =========================================================

def simple_embedding(text):
    vec = [hash(word) % 1000 for word in text.split()[:128]]
    return np.array(vec, dtype=float)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

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

    return chat, collection, memory


chat, collection, memory = load_models()


# =========================================================
# LOAD PDF FROM LOCAL FILE
# =========================================================

def load_pdf_text(file_path):
    reader = PdfReader(file_path)
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
# LOAD PDF INTO VECTOR DB
# =========================================================

def load_pdf_to_db():

    if collection.count() > 0:
        return

    if not os.path.exists(PDF_FILE):
        st.error(f"{PDF_FILE} not found in repo")
        return

    text = load_pdf_text(PDF_FILE)

    if not text.strip():
        st.error("PDF contains no readable text")
        return

    chunks = chunk_text(text)

    embeddings = [simple_embedding(c).tolist() for c in chunks]

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )


load_pdf_to_db()


# =========================================================
# RETRIEVE CONTEXT
# =========================================================

def retrieve_context(query):

    if collection.count() == 0:
        return ["No documents available"]

    q_embed = simple_embedding(query).tolist()

    results = collection.query(
        query_embeddings=[q_embed],
        n_results=3
    )

    return results["documents"][0]


# =========================================================
# ASK AI
# =========================================================

def ask_ai(question):

    system_prompt = """
You are an AI clone of Rahul Pakhare, a GIS consultant.
Speak professionally and concisely.
Answer only from provided context.
If answer not found, say "Not available in my data".
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
