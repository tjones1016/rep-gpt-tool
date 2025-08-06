import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load OpenAI key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Rep GPT Tool")
st.title("📣 Pro-Roofing AI Sales Assistant")

@st.cache_resource
def setup_qa_chain():
    loaders = [
        Docx2txtLoader("data/sales_guide.docx"),
        PyPDFLoader("data/d2d_script.pdf")
    ]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    if os.path.exists("vectorstore/index.faiss"):
        db = FAISS.load_local("vectorstore", OpenAIEmbeddings())
    else:
        db = FAISS.from_documents(split_docs, OpenAIEmbeddings())
        db.save_local("vectorstore")

    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
