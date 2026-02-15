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
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    # Chat Model
    chat = ChatGroq(
        temperature=0.3,
        model="llama-3.3-70b-versatile",  # update as per available Groq model
        groq_api_key=GROQ_API_KEY
    )

    # Embedding Model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Chroma client and collection
    chroma_client = chromadb.Client(
        chromadb.Settings(
            persist_directory=CHROMA_DB_DIR,
            anonymized_telemetry=False
        )
    )
    collection = chroma_client.get_or_create_collection(
        name="knowledge",
        embedding_function=embedding_model.embed_documents
    )

    # Conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return chat, embedding_model, collection, memory

chat, embedding_model, collection, memory = load_models()

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
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# =========================================================
# SPLIT TEXT INTO CHUNKS
# =========================================================
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,   # larger chunks
        chunk_overlap=300  # overlap for context
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
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embedding_model.embed_documents(chunks)
    )

# Load PDF automatically
load_pdf_to_db(PDF_FILE)

# =========================================================
# RETRIEVE CONTEXT FOR QUERY
# =========================================================
def retrieve_context(query):
    if collection.count() == 0:
        return "No documents loaded."

    results = collection.query(
        query_embeddings=[embedding_model.embed_query(query)],
        n_results=3
    )
    context_chunks = results["documents"][0]
    return " ".join(context_chunks)

# =========================================================
# ASK AI
# =========================================================
def ask_ai(question):
    system_prompt = """
You are Rahul Pakhare's AI clone.
Answer professionally using only the provided context.
Do not hallucinate. Use human-like tone.
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

# Display chat history
for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
question = st.chat_input("Ask something...")

if question:
    st.session_state.history.append({"role": "user", "content": question})

    with st.spinner("Thinking..."):
        answer = ask_ai(question)

    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()
