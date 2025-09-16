import os
import re
from typing import Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Rep GPT — Pro-Roofing Sales Assistant",
    page_icon="apple-touch-icon.png",
    layout="wide"
)

st.markdown("""
<link rel="apple-touch-icon" sizes="180x180" href="apple-touch-icon.png">
""", unsafe_allow_html=True)

# -----------------------------
# LangChain + RAG
# -----------------------------
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from PyPDF2 import PdfReader
import docx2txt

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("📣 Pro-Roofing AI Sales Assistant")

# -----------------------------
# Load docs from /data
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
# Hardcoded Sales Knowledge
# -----------------------------
HARDCODED_KNOWLEDGE = {
    "slap": (
        "The SLAP formula:\n"
        "1. Say hi and introduce yourself\n"
        "2. Let them know why you're here\n"
        "3. Ask an open-ended question\n"
        "4. Present your pitch"
    ),
    "aro": (
        "The ARO formula:\n"
        "1. Acknowledge — validate the homeowner's concern "
        "(e.g. 'I hear you, that sounds stressful')\n"
        "2. Reassure — provide confident reassurance "
        "(e.g. 'We'll walk you through this')\n"
        "3. Overcome — pivot to the next step (inspection, filing the claim, referral, or close)"
    ),
    "pricing_payment_accountability": (
        "After filing the claim, the rep should explain **Pricing, Payment, and Accountability**:\n"
        "- **Pricing**: Adjusters use Xactimate to price the claim.\n"
        "- **Payment**: Typical flow — 1st payment (initial/ACV), homeowner deductible, "
        "then 2nd payment (recoverable depreciation/final check).\n"
        "- **Accountability**: The carrier controls the process; there is no money 'saved' "
        "by choosing a cheaper contractor. Use the 'body shop' analogy to explain how "
        "insurance sets the amount and the contractor performs the repair."
    ),
    "file_claim": (
        "If you're not sure whether there’s enough damage on a roof, always end with:\n"
        "\"File the Claim — let the adjuster make the call.\""
    )
}

def inject_hardcoded_answers(user_input: str) -> Optional[str]:
    text = user_input.lower().strip()

    if re.search(r"\baro\b", text) or "what is aro" in text or "what's aro" in text:
        return HARDCODED_KNOWLEDGE["aro"]

    if "slap" in text or "what is slap" in text or "what's slap" in text:
        return HARDCODED_KNOWLEDGE["slap"]

    if any(k in text for k in ("pricing", "payment", "accountability", "xactimate", "deductible", "recoverable depreciation")):
        return HARDCODED_KNOWLEDGE["pricing_payment_accountability"]

    if re.search(r"(not sure|unsure|borderline|maybe).*damage", text) or re.search(r"enough damage", text):
        return HARDCODED_KNOWLEDGE["file_claim"]

    return None

def needs_file_claim_append(user_input: str) -> bool:
    """Check if the query mentions uncertainty about roof damage — to append 'File the Claim' reminder."""
    text = user_input.lower()
    return bool(re.search(r"(not sure|unsure|borderline|maybe).*damage", text) or "enough damage" in text)

# -----------------------------
# Chat UI
# -----------------------------
st.caption("Ask about sales, pay, objections, pricing, or process steps. Powered by your `/data` docs + built-in knowledge.")

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

    # Hardcoded answers first
    hardcoded = inject_hardcoded_answers(prompt)
    if hardcoded:
        result = hardcoded
    else:
        result = qa_chain.run({"question": prompt, "chat_history": st.session_state.chat_history})
        # Safety net: append file-claim guidance if the question suggests uncertainty
        if needs_file_claim_append(prompt):
            result += "\n\n" + HARDCODED_KNOWLEDGE["file_claim"]

    st.chat_message("assistant").markdown(result)
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.session_state.chat_history.append((prompt, result))
