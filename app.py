import os
import re
import io
import base64
from typing import Dict, Any, Tuple

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# -----------------------------
# Page config + favicon / app icons
# -----------------------------
logo_path = os.path.join("data", "logo.png")

if os.path.exists(logo_path):
    st.set_page_config(
        page_title="Rep GPT — Chat + Estimates",
        page_icon=logo_path,
        layout="wide"
    )

    # Embed as Apple Touch Icon so iOS home screen sees it
    with open(logo_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <link rel="apple-touch-icon" sizes="180x180" href="data:image/png;base64,{b64}">
        <link rel="icon" type="image/png" sizes="32x32" href="data:image/png;base64,{b64}">
        <link rel="icon" type="image/png" sizes="16x16" href="data:image/png;base64,{b64}">
        """,
        unsafe_allow_html=True,
    )
else:
    st.set_page_config(
        page_title="Rep GPT — Chat + Estimates",
        page_icon="📣",
        layout="wide"
    )

st.title("📣 Pro-Roofing AI Sales Assistant")

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("⚠️ Missing OpenAI API Key. Please set it in your environment.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Document ingestion
# -----------------------------
@st.cache_resource
def load_docs_and_create_index():
    docs = []
    data_folder = "data"
    if os.path.exists(data_folder):
        for file in os.listdir(data_folder):
            if file.endswith(".pdf"):
                pdf_reader = PdfReader(os.path.join(data_folder, file))
                for page in pdf_reader.pages:
                    docs.append(page.extract_text())

    if not docs:
        return None

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(docs, embedding=embeddings)
    return vectorstore

vectorstore = load_docs_and_create_index()

# -----------------------------
# Prompt Template
# -----------------------------
template = """You are Rep GPT, a sales assistant trained on Pro-Roofing’s material. 
Answer clearly, concisely, and always stay professional. 
If unsure, say you’re unsure rather than making up an answer.

{context}

Question: {question}
Answer:"""

QA_PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

# -----------------------------
# Conversational Chain
# -----------------------------
def setup_conversational_chain():
    if not vectorstore:
        return None
    return ConversationalRetrievalChain.from_llm(
        llm=client.chat.completions,
        retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )

qa_chain = setup_conversational_chain()

# -----------------------------
# Chat UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask me something about Pro-Roofing…"):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if qa_chain:
            result = qa_chain(
                {"question": user_input, "chat_history": st.session_state["messages"]}
            )
            response = result["answer"]
        else:
            response = "⚠️ No training docs loaded. Please upload PDFs."

        st.markdown(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})
