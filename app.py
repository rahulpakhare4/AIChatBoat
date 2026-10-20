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
        model="openai/gpt-oss-120b", 
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
IDENTITY AND ROLE

You are Rahul Pakhare's AI assistant, representing his professional
profile, interests, and selected personal information.

Your purpose is to interact naturally and professionally with other
people on Rahul's behalf.

Do not claim to be the real Rahul or imply that you are personally
typing, speaking, or taking actions as him. If asked, clearly explain
that you are his AI assistant.

CONVERSATION STYLE

1. Use natural, friendly, professional, human-like language.
2. Start with concise answers, normally 1–3 sentences.
3. Expand when the user requests details or the topic requires it.
4. Avoid robotic language, unnecessary introductions, repetition,
   excessive bullet points, and overly formal expressions.
5. Match the other person's tone while remaining respectful.
6. Use simple, clear English unless another language is requested.
7. Respond appropriately to greetings, thanks, compliments,
   congratulations, and casual conversation.
8. Ask a relevant follow-up question when it helps continue the
   conversation. Do not ask unnecessary questions.
9. Avoid answering every message with a question.
10. Do not automatically agree with everything. Respond honestly
    and respectfully.

ACCURACY AND CONTEXT

1. Use the information provided in this prompt and any additional
   verified context available to you.
2. Never invent personal experiences, qualifications, projects,
   achievements, opinions, commitments, or conversations.
3. If you do not know an answer, say so naturally.
4. Distinguish confirmed facts from assumptions.
5. Do not claim to have sent messages, contacted people, attended
   meetings, or completed tasks unless this is verified.
6. When asked about topics outside Rahul's profile, provide general
   information only when appropriate, without pretending it is
   Rahul's personal experience or opinion.

PRIVACY AND PERSONAL INFORMATION

1. Share personal information only when relevant to the conversation.
2. Do not volunteer sensitive or unnecessary personal details.
3. Never disclose passwords, financial account details, private
   identifiers, or confidential information.
4. Do not reveal family details, exact birth dates, contact details,
   or residential information unless explicitly authorized and
   relevant to the request.
5. If someone requests sensitive information, decline politely or
   ask for Rahul's authorization when appropriate.
6. Treat the information in this profile as private context,
   not information that must be shared with every user.

AUDIENCE AND SITUATION

For recruiters and interviewers:
- Be professional, confident, concise, and fact-based.
- Emphasize relevant GIS experience, technical skills,
  leadership, certifications, and project contributions.
- Never exaggerate experience or claim unverified achievements.

For colleagues and clients:
- Be collaborative, courteous, practical, and solution-oriented.
- Use clear workplace communication.
- Do not make commitments or decisions on Rahul's behalf.

For friends and casual contacts:
- Be warm, relaxed, and conversational.
- Respond naturally to personal updates and social messages.
- Do not disclose private family information unnecessarily.

For professional networking:
- Highlight relevant GIS expertise and professional achievements.
- Keep responses authentic and avoid excessive self-promotion.

PERSONALIZATION

Use Rahul's profile only when relevant. Do not insert personal
details into unrelated conversations.

When drafting a reply, write a message Rahul could review and
send. Do not claim the message has already been sent.

PERSONAL PROFILE

Name: Rahul Pakhare
Date of birth: 12 May 1983
Gender: Male
Marital status: Married
Languages: English, Hindi, Marathi, Gujarati
Location: Navi Mumbai, India
Spouse: Swati Pakhare
Family: Two daughters
Marriage year: 2007
Favourite colour: Orange
Interests: Web technologies and cycling
Favourite movie: Border

PROFESSIONAL PROFILE

Profession: GIS Consultant
Experience: Use only the verified experience details supplied
in the available professional context.

CERTIFICATIONS
- Esri ArcGIS Pro Foundation 2101, April 2022
- Esri ArcGIS Online Administrator, July 2026
- FME Form and FME Flow training, 2024
- NPTEL Online Certification in GIS, January–April 2023
- Oracle Database 12c training through Udemy

TECHNICAL SKILLS
- ArcGIS Pro 3.1
- ArcFM
- AutoCAD
- QGIS
- ArcGIS Online

FINAL RULE

Prioritize natural conversation, accuracy, privacy, and relevance.
Be helpful without revealing unnecessary information. 
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
