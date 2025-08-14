import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Rep GPT Chat", layout="wide")
st.title("📣 Pro-Roofing AI Sales Assistant (Chat Mode)")

# Function to load all files in /data folder
def load_all_docs():
    docs = []
    for filename in os.listdir("data"):
        path = os.path.join("data", filename)
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

# Initialize chain and memory
qa_chain = setup_conversational_chain()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask about sales, pay, objections, pricing, or process steps..."):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run chain with history
    result = qa_chain.run({"question": prompt, "chat_history": st.session_state.chat_history})
    st.chat_message("assistant").markdown(result)
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.session_state.chat_history.append((prompt, result))
