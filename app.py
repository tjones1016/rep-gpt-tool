import streamlit as st
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import openai
import os

# Page setup
st.set_page_config(page_title="📣 Pro-Roofing AI Sales Assistant")
st.title("📣 Pro-Roofing AI Sales Assistant")
st.caption("Ask a question and the assistant will answer using your training documents.")

# API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to load and process documents
@st.cache_resource
def setup_qa_chain():
    loaders = []
    error_messages = []

    # Attempt to load both documents
    if os.path.exists("data/sales_guide.docx"):
        loaders.append(Docx2txtLoader("data/sales_guide.docx"))
    else:
        error_messages.append("Missing: sales_guide.docx")

    if os.path.exists("data/d2d_script.pdf"):
        loaders.append(PyMuPDFLoader("data/d2d_script.pdf"))
    else:
        error_messages.append("Missing: d2d_script.pdf")

    if error_messages:
        st.warning("⚠️ Some files are missing:\n\n" + "\n".join(error_messages))

    documents = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"❌ Error loading {loader}: {str(e)}")
