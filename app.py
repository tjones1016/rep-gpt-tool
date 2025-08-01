import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load your OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Rep GPT Tool")
st.title("📣 Pro-Roofing AI Sales Assistant")

# Load and embed documents
@st.cache_resource
def setup_qa_chain():
    loaders = [
        Docx2txtLoader("data/Sales Guide.docx"),
        PyPDFLoader("data/D2D Conversion Script.pdf")
    ]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # Chunk the documents for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # Create or load the vector store
    if os.path.exists("vectorstore/index.faiss"):
        db = FAISS.load_local("vectorstore", OpenAIEmbeddings())
    else:
        db = FAISS.from_documents(split_docs, OpenAIEmbeddings())
        db.save_local("vectorstore")

    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    retriever = db.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Set up the RAG chain
qa_chain = setup_qa_chain()

# User input field
query = st.text_input("Ask a question about sales, scripts, objections, or processes:", "")

if query:
    with st.spinner("Searching your training materials..."):
        answer = qa_chain.run(query)
        st.markdown(f"### 🧠 Answer:\n{answer}")
