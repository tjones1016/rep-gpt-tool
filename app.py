import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OPENAI_API_KEY is missing. Please set it in your environment variables.")
    st.stop()

# -----------------------------
# Hardcoded company rules
# -----------------------------
HARD_CODED_RULES = {
    "slap": """SLAP Formula:
1. Say hi and introduce yourself
2. Let them know why you’re here
3. Ask an open-ended question
4. Present your pitch""",

    "aro": """ARO Formula:
1. Acknowledge
2. Reassure
3. Overcome""",

    "pricing_payment_accountability": """After filing the claim, the rep must explain:
- Pricing: Adjusters use Xactimate
- Payment: 1st payment (ACV), deductible, 2nd payment (depreciation)
- Accountability: No money is saved, carrier controls the process.
Use the body shop analogy to communicate the simplicity of the process.""",

    "file_claim": """If you’re not sure whether there’s enough roof damage, always advise:
'File the Claim — let the adjuster make the call.'"""
}

# -----------------------------
# Function to load documents and build vectorstore
# -----------------------------
@st.cache_resource
def load_vectorstore():
    data_dir = "data"
    docs = []

    for file in os.listdir(data_dir):
        filepath = os.path.join(data_dir, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            docs.extend(loader.load())
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
            docs.extend(loader.load())
        elif file.endswith(".csv"):
            loader = CSVLoader(filepath)
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_documents(split_docs, embeddings)


vectorstore = load_vectorstore()

# -----------------------------
# Conversational Chain Setup
# -----------------------------
def setup_conversational_chain():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=openai_api_key
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )


qa_chain = setup_conversational_chain()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Pro-Roofing AI Assistant",
    page_icon=":construction:"
)

st.title("🏠 Pro-Roofing AI Assistant")
st.write("Ask me about sales scripts, pricing, storm damage, or company processes.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your question here...")

if user_input:
    lower_q = user_input.lower()

    # --- Hardcoded rule checks ---
    if "slap" in lower_q:
        response = HARD_CODED_RULES["slap"]
    elif "aro" in lower_q:
        response = HARD_CODED_RULES["aro"]
    elif any(word in lower_q for word in ["pricing", "payment", "accountability"]):
        response = HARD_CODED_RULES["pricing_payment_accountability"]
    elif "not sure" in lower_q or "enough damage" in lower_q:
        response = HARD_CODED_RULES["file_claim"]
    else:
        result = qa_chain({"question": user_input})
        response = result["answer"]

    # Save chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AI", response))

# --- Display chat history ---
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
