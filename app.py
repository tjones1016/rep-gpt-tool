import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- CONFIG ---
st.set_page_config(page_title="Pro Roofing AI Assistant", page_icon="apple-touch-icon.png")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in environment variables.")
    st.stop()

# --- LOAD DOCUMENTS ---
@st.cache_resource
def load_vectorstore():
    docs = []

    data_files = [
        "data/d2d_script.pdf",
        "data/sales_guide.docx",
        "data/sales_guide3.docx",
        "data/pricing.csv",
    ]

    for filepath in data_files:
        if not os.path.exists(filepath):
            st.warning(f"File not found: {filepath}")
            continue

        if filepath.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filepath.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        elif filepath.endswith(".csv"):
            loader = CSVLoader(filepath)
        else:
            st.warning(f"Unsupported file type: {filepath}")
            continue

        docs.extend(loader.load())

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(chunks, embeddings)

vectorstore = load_vectorstore()

# --- SETUP CHAIN ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
    retriever=vectorstore.as_retriever(),
    memory=memory,
)

# --- HARDCODED RULES ---
def apply_rules(user_input: str):
    rules = {
        "slap": "Remember SLAP: Smile, Look, Ask, Pause.",
        "aro": "Use ARO: Acknowledge, Reassure, Overcome.",
        "pricing": "Pricing must follow the official pricing guide in pricing.csv.",
        "payment": "Payment and accountability rules are outlined in the Sales Guide.",
        "file the claim": "If unsure, fallback to: 'Let’s go ahead and file the claim with the insurance provider.'",
    }
    for key, response in rules.items():
        if key in user_input.lower():
            return response
    return None

# --- UI ---
st.title("🤖 Pro Roofing AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if user_input := st.chat_input("Ask me something..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Check hard-coded rules
    rule_response = apply_rules(user_input)
    if rule_response:
        response = rule_response
    else:
        result = qa_chain({"question": user_input})
        response = result["answer"]

    # Store and display response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
