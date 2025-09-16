import os
from typing import Dict, Any

import streamlit as st
from dotenv import load_dotenv

# -----------------------------
# Apple Touch Icon for iPhone
# -----------------------------
st.set_page_config(
    page_title="Rep GPT — Chat",
    page_icon="apple-touch-icon.png",
    layout="wide"
)

st.markdown("""
<link rel="apple-touch-icon" sizes="180x180" href="apple-touch-icon.png">
""", unsafe_allow_html=True)

# LangChain + RAG bits
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("📣 Pro-Roofing AI Sales Assistant")

# -----------------------------
# RAG: load all docs in /data
# -----------------------------
def load_all_docs():
    docs = []
    data_dir = "data"
    if not os.path.isdir(data_dir):
        return docs
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        if filename.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif filename.lower().endswith(".docx"):
            docs.extend(Docx2txtLoader(path).load())
        elif filename.lower().endswith(".txt"):
            docs.extend(TextLoader(path).load())
    return docs

@st.cache_resource
def setup_conversational_chain():
    docs = load_all_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    split_docs = splitter.split_documents(docs)

    if os.path.exists("vectorstore/index.faiss"):
        db = FAISS.load_local(
            "vectorstore",
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        )
    else:
        db = FAISS.from_documents(split_docs, OpenAIEmbeddings())
        db.save_local("vectorstore")

    retriever = db.as_retriever()
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    return ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

qa_chain = setup_conversational_chain()

# -----------------------------
# Chat UI
# -----------------------------
st.caption("Ask about sales, objections, SLAP/ARO, pay, pricing, or process steps. Powered by your `/data` docs.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your question…"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    result = qa_chain.run({"question": prompt, "chat_history": st.session_state.chat_history})
    st.chat_message("assistant").markdown(result)
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.session_state.chat_history.append((prompt, result))
