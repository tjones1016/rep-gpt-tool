import os
import glob
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader

# ---------------------
# Streamlit Page Config
# ---------------------
st.set_page_config(
    page_title="Pro-Roofing AI Assistant",
    page_icon="apple-touch-icon.png"
)

st.title("🦾 Pro-Roofing AI Assistant")
st.caption("Powered by LangChain + OpenAI")

# ---------------------
# Load Environment Vars
# ---------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found in environment variables.")
    st.stop()

# ---------------------
# Optional DOCX loader
# ---------------------
try:
    from langchain.document_loaders import UnstructuredWordDocumentLoader
    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False
    st.warning(
        "⚠️ `unstructured` package not installed. .docx files will be skipped."
    )

# ---------------------
# Optional PDF loader
# ---------------------
try:
    from langchain.document_loaders import PyPDFLoader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    st.warning(
        "⚠️ `PyPDF2` package not installed. PDF files will be skipped."
    )

# ---------------------
# Load all docs from data/
# ---------------------
@st.cache_resource
def load_vectorstore():
    docs = []

    for filepath in glob.glob("data/*"):
        if filepath.endswith(".txt"):
            loader = TextLoader(filepath)
            docs.extend(loader.load())
        elif filepath.endswith(".docx"):
            if HAS_UNSTRUCTURED:
                loader = UnstructuredWordDocumentLoader(filepath)
                docs.extend(loader.load())
            else:
                st.info(f"Skipping {filepath}: unstructured not installed.")
        elif filepath.endswith(".pdf"):
            if HAS_PDF:
                loader = PyPDFLoader(filepath)
                docs.extend(loader.load())
            else:
                st.info(f"Skipping {filepath}: PyPDF2 not installed.")
        else:
            st.info(f"Skipping unsupported file: {filepath}")

    if not docs:
        st.error("❌ No documents could be loaded from the data folder.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = load_vectorstore()

# ---------------------
# Conversational Chain
# ---------------------
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever()
)

# ---------------------
# Chat UI
# ---------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Ask me anything about roofing, sales process, or materials...")

if user_query:
    result = qa_chain(
        {"question": user_query, "chat_history": st.session_state.chat_history}
    )
    st.session_state.chat_history.append((user_query, result["answer"]))

# Display Chat
for i, (q, a) in enumerate(st.session_state.chat_history):
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)
