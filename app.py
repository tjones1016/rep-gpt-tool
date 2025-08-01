import streamlit as st
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import openai

# 🔑 Load API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 🧠 Build RAG pipeline with LangChain
@st.cache_resource
def setup_qa_chain():
    # Load documents
    loaders = [
        Docx2txtLoader("data/sales_guide.docx"),
        PyMuPDFLoader("data/d2d_script.pdf")
    ]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and store in FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Create Retrieval QA chain
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🔵 Streamlit UI
st.set_page_config(page_title="📣 Pro-Roofing AI Sales Assistant")
st.title("📣 Pro-Roofing AI Sales Assistant")
st.caption("Ask a question and the assistant will answer using your training documents.")

user_input = st.text_area("What do you want to ask?", height=100)
qa_chain = setup_qa_chain()

if st.button("Ask") and user_input.strip():
    with st.spinner("Thinking..."):
        result = qa_chain.run(user_input)
        st.markdown("### 💬 Response")
        st.write(result)
