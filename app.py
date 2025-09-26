import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain

# ---------------------
# Streamlit Page Config
# ---------------------
st.set_page_config(
    page_title="Pro-Roofing AI Assistant",
    page_icon="apple-touch-icon.png"  # ✅ Uses your logo for tab + favicon
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
# Build Vector DB (example with sales guide)
# ---------------------
@st.cache_resource
def load_vectorstore():
    # You can swap this for multiple docs later
    loader = TextLoader("data/sales_guide.docx")  # adjust if needed
    docs = loader.load()

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
