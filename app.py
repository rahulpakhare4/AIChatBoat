import os
import sys
import shutil
import streamlit as st

# ================================
# SQLITE FIX (Streamlit Cloud)
# ================================
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================================================
# CONFIG
# =========================================================
PDF_FILE = "Rahul Pakhare.pdf"
CHROMA_DB_DIR = "./chroma_db"

# =========================================================
# LOAD API KEY
# =========================================================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in secrets.toml")
    st.stop()

# =========================================================
# RESET CHROMA DB IF EXISTS
# =========================================================
if os.path.exists(CHROMA_DB_DIR):
    shutil.rmtree(CHROMA_DB_DIR)

# =========================================================
# SIMPLE EMBEDDING FUNCTION
# =========================================================
def simple_embedding(text, dim=128):
    vec = [0] * dim
    words = text.split()
    for i in range(min(len(words), dim)):
        vec[i] = hash(words[i]) % 1000
    return vec

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    chat = ChatGroq(
        temperature=0.3,
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY
    )
    chroma_client = chromadb.Client(
        chromadb.Settings(
            persist_directory=CHROMA_DB_DIR,
            anonymized_telemetry=False
        )
    )
    collection = chroma_client.get_or_create_collection(
        name="knowledge",
        metadata={"hnsw:space": "cosine"}
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return chat, collection, memory

chat, collection, memory = load_models()

# =========================================================
# LOAD PDF
# =========================================================
def read_pdf(file_path):
    if not os.path.exists(file_path):
        st.error(f"{file_path} not found in repo")
        return ""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# =========================================================
# SPLIT TEXT INTO CHUNKS
# =========================================================
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# =========================================================
# LOAD PDF INTO CHROMA
# =========================================================
def load_pdf_to_db(pdf_file):
    if collection.count() > 0:
        return

    text = read_pdf(pdf_file)
    if not text.strip():
        st.warning(f"{pdf_file} has no extractable text!")
        return

    chunks = split_text(text)
    embeddings = [simple_embedding(c) for c in chunks]

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )

load_pdf_to_db(PDF_FILE)

# =========================================================
# RETRIEVE CONTEXT FOR QUERY
# =========================================================
def retrieve_context(query):
    if collection.count() == 0:
        return "No documents loaded."

    q_embed = simple_embedding(query)
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=3
    )
    
    # documents[0] is already a string
    if results["documents"]:
        return results["documents"][0]
    return "No relevant context found."

# =========================================================
# ASK AI
# =========================================================
def ask_ai(question):
    system_prompt = """
You are Rahul Pakhare's AI clone.
Answer professionally using only the provided context.
Do not hallucinate. Use human-like tone.
PERSONAL DETAILS 
Name: Rahul Pakhare 
DOB: 12th May 1983 
Gender: Male 
Marital Status: Married 
Languages Known: English, Hindi, Marathi, Gujarati 
Address: Navi Mumbai 
Spouse: Swati  Pakhare 
Elder daughter : 13 Years old 
Younger:  1 Years old 
Marriage date: 2007 
Favourite colour: Orange 
Weight : 63 KG 
Height: 172 cm 
Interest : Web technologies, Cycling. 
Favourite movie : Border 
CERTIFICATION 
• ESRI ArcGIS Pro Foundation 2101(Apr-2022) 
• FME Flow, FME Form training course (2024) 
• NPTEL Online Certification in GIS (Jan-Apr 2023) 
• Oracle Database 12C from Udemy 
SKILLS 
GIS Software: ArcGIS Pro 3.1, ARCFM, AutoCAD, QGIS, ArcGIS Online, 
"""
    context = retrieve_context(question)
    history = memory.load_memory_variables({}).get("chat_history", [])[-6:]
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"History:{history}\nContext:{context}\nQ:{question}")
    ]
    try:
        response = chat.invoke(messages)
        answer = response.content
    except Exception as e:
        answer = f"Model error: {e}"

    memory.save_context({"input": question}, {"output": answer})
    return answer

# =========================================================
# STREAMLIT UI
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
