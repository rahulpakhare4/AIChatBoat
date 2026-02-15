import os
import sys
import streamlit as st

# =========================================================
# SQLITE FIX FOR STREAMLIT CLOUD
# =========================================================
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except:
    pass


# =========================================================
# IMPORTS
# =========================================================
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
PDF_PATH = "Rahul Pakhare.pdf"


# =========================================================
# LOAD API KEY
# =========================================================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in Streamlit secrets.")
    st.stop()


# =========================================================
# LOAD MODELS (CACHED)
# =========================================================
@st.cache_resource
def load_models():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = ChatGroq(
        temperature=0.3,
        model="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY
    )

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection("knowledge")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return embeddings, llm, collection, memory


embedding_model, chat, collection, memory = load_models()


# =========================================================
# LOAD PDF + CREATE VECTOR DB
# =========================================================
@st.cache_resource
def load_pdf():

    if collection.count() > 0:
        return "Already Loaded"

    if not os.path.exists(PDF_PATH):
        return "PDF NOT FOUND"

    reader = PdfReader(PDF_PATH)

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )

    chunks = splitter.split_text(text)

    embeddings = embedding_model.embed_documents(chunks)

    collection.add(
        ids=[f"id_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )

    return "Loaded"


status = load_pdf()

if status == "PDF NOT FOUND":
    st.error("Rahul Pakhare.pdf not found in repo folder.")
    st.stop()


# =========================================================
# RETRIEVER
# =========================================================
def retrieve_context(query):

    if collection.count() == 0:
        return ["No knowledge available"]

    q_embed = embedding_model.embed_query(query)

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
You are an AI clone of Rahul Pakhare, a GIS Consultant.

Rules:
- Answer professionally
- Be concise
- Use provided context only
- If answer not found, say: "I don't have information about that."
"""

    context = retrieve_context(question)
    history = memory.load_memory_variables({}).get("chat_history", [])[-6:]

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"""
Conversation History:
{history}

Context:
{context}

User Question:
{question}
"""
        )
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
st.set_page_config(page_title="Rahul AI Clone", page_icon="🤖")

st.title("🤖 Rahul AI Clone")

if "history" not in st.session_state:
    st.session_state.history = []

# show chat history
for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

# input
question = st.chat_input("Ask something about Rahul...")

if question:

    st.session_state.history.append({"role": "user", "content": question})

    with st.spinner("Thinking..."):
        answer = ask_ai(question)

    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()
