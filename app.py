import os
import re
import io
from typing import Dict, Any, Tuple

import streamlit as st
from dotenv import load_dotenv

# -----------------------------
# NEW: Apple Touch Icon for iPhone
# -----------------------------
# Make sure you have `apple-touch-icon.png` in the same folder as this app.py
st.set_page_config(
    page_title="Rep GPT — Chat + Estimates",
    page_icon="apple-touch-icon.png",  # your logo file
    layout="wide"
)

# Optional: enforce Apple Touch Icon in HTML (extra safety for iPhones)
st.markdown("""
<link rel="apple-touch-icon" sizes="180x180" href="apple-touch-icon.png">
""", unsafe_allow_html=True)

# LangChain + RAG bits
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# File parsing
from PyPDF2 import PdfReader
import docx2txt

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("📣 Pro-Roofing AI Sales Assistant")

# -----------------------------
# RAG: load all docs in /data
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

    # Vector store cache
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
# EagleView parsing
# -----------------------------
def parse_eagleview_text(text: str) -> Dict[str, float]:
    t = text.lower().replace(",", "")
    data: Dict[str, float] = {
        "total_squares": 0.0,
        "total_area_sqft": 0.0,
        "ridge_lf": 0.0,
        "hip_lf": 0.0,
        "valley_lf": 0.0,
        "eave_lf": 0.0,
        "dripedge_lf": 0.0,
    }

    m_sq = re.search(r"total squares\D*([0-9]+(?:\.[0-9]+)?)", t)
    if m_sq:
        data["total_squares"] = float(m_sq.group(1))

    m_area = re.search(r"(total (?:roof )?area|roof area)\D*([0-9]+(?:\.[0-9]+)?)\s*(sq\s*ft|square feet|sf)", t)
    if m_area:
        data["total_area_sqft"] = float(m_area.group(2))

    def grab_linear(label: str, key: str):
        m = re.search(rf"{label}\D*([0-9]+(?:\.[0-9]+)?)\s*(?:lf|linear feet?|lineal feet?)", t)
        if m:
            data[key] = float(m.group(1))

    grab_linear("ridge", "ridge_lf")
    grab_linear("hip", "hip_lf")
    grab_linear("valley", "valley_lf")
    grab_linear("eave", "eave_lf")
    grab_linear("drip edge", "dripedge_lf")

    return data

def read_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

# -----------------------------
# Estimate math
# -----------------------------
def compute_estimate(inputs: Dict[str, float], pricing: Dict[str, float], waste_pct: float = 10.0, op_pct: float = 20.0) -> Tuple[Dict[str, Any], float, float]:
    squares = inputs.get("total_squares", 0.0)
    if squares <= 0:
        sqft = inputs.get("total_area_sqft", 0.0)
        if sqft > 0:
            squares = sqft / 100.0

    adj_squares = squares * (1.0 + waste_pct / 100.0)

    def p(key: str) -> float:
        return float(pricing.get(key, 0.0))

    line_items: Dict[str, Any] = {}

    shingles_cost = adj_squares * p("shingles_per_square")
    line_items["Shingles (with waste)"] = {"qty": round(adj_squares, 2), "unit": "sq", "rate": p("shingles_per_square"), "cost": round(shingles_cost, 2)}

    for label, key_in, key_price in [
        ("Ridge", "ridge_lf", "ridge_per_lf"),
        ("Hip", "hip_lf", "hip_per_lf"),
        ("Valley", "valley_lf", "valley_per_lf"),
        ("Eave Starter", "eave_lf", "eave_per_lf"),
        ("Drip Edge", "dripedge_lf", "dripedge_per_lf"),
    ]:
        lf = float(inputs.get(key_in, 0.0))
        rate = p(key_price)
        if lf > 0 and rate > 0:
            line_items[label] = {"qty": lf, "unit": "lf", "rate": rate, "cost": round(lf * rate, 2)}

    subtotal = round(sum(item["cost"] for item in line_items.values()), 2)
    total_with_op = round(subtotal * (1 + op_pct / 100.0), 2)

    return line_items, subtotal, total_with_op

# -----------------------------
# UI
# -----------------------------
tab_chat, tab_est = st.tabs(["💬 Chat", "📐 Estimate Calculator"])

# ----- Chat tab -----
with tab_chat:
    st.caption("Ask about sales, pay, objections, pricing, or process steps. Powered by your `/data` docs.")
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

        result = qa_chain.run({"question": prompt, "chat_history": st.session_state.chat_history})
        st.chat_message("assistant").markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.session_state.chat_history.append((prompt, result))

# ----- Estimate tab -----
with tab_est:
    st.subheader("Estimate Calculator (10% waste applied automatically)")

    with st.expander("Edit pricing", expanded=False):
        colA, colB, colC = st.columns(3)
        shingles_per_square = colA.number_input("Shingles ($/square)", value=435.0, min_value=0.0, step=1.0)
        ridge_per_lf = colB.number_input("Ridge ($/LF)", value=5.0, min_value=0.0, step=0.5)
        hip_per_lf = colC.number_input("Hip ($/LF)", value=5.0, min_value=0.0, step=0.5)
        valley_per_lf = colA.number_input("Valley ($/LF)", value=8.0, min_value=0.0, step=0.5)
        eave_per_lf = colB.number_input("Eave Starter ($/LF)", value=1.5, min_value=0.0, step=0.5)
        dripedge_per_lf = colC.number_input("Drip Edge ($/LF)", value=1.5, min_value=0.0, step=0.5)

    pricing = {
        "shingles_per_square": shingles_per_square,
        "ridge_per_lf": ridge_per_lf,
        "hip_per_lf": hip_per_lf,
        "valley_per_lf": valley_per_lf,
        "eave_per_lf": eave_per_lf,
        "dripedge_per_lf": dripedge_per_lf,
    }

    col1, col2 = st.columns(2)

    with col1:
        ev_file = st.file_uploader("Upload EagleView PDF", type=["pdf"])
        parsed_inputs = {}
        if ev_file is not None:
            try:
                pdf_text = read_pdf_text(ev_file.read())
                parsed_inputs = parse_eagleview_text(pdf_text)
                st.success("Parsed EagleView values from PDF.")
                st.json({k: round(v, 2) for k, v in parsed_inputs.items() if v})
            except Exception as e:
                st.error(f"Could not read PDF: {e}")

    with col2:
        st.caption("Manual override / entry")
        total_squares = st.number_input("Total Squares", value=float(parsed_inputs.get("total_squares", 0.0)), min_value=0.0, step=0.1)
        total_area_sqft = st.number_input("Total Area (sq ft)", value=float(parsed_inputs.get("total_area_sqft", 0.0)), min_value=0.0, step=10.0)
        ridge_lf = st.number_input("Ridge (LF)", value=float(parsed_inputs.get("ridge_lf", 0.0)), min_value=0.0, step=1.0)
        hip_lf = st.number_input("Hip (LF)", value=float(parsed_inputs.get("hip_lf", 0.0)), min_value=0.0, step=1.0)
        valley_lf = st.number_input("Valley (LF)", value=float(parsed_inputs.get("valley_lf", 0.0)), min_value=0.0, step=1.0)
        eave_lf = st.number_input("Eave Starter (LF)", value=float(parsed_inputs.get("eave_lf", 0.0)), min_value=0.0, step=1.0)
        dripedge_lf = st.number_input("Drip Edge (LF)", value=float(parsed_inputs.get("dripedge_lf", 0.0)), min_value=0.0, step=1.0)

        waste_pct = st.slider("Waste %", min_value=0, max_value=25, value=10, step=1)
        op_pct = st.slider("Overhead & Profit %", min_value=0, max_value=30, value=20, step=1)

        if st.button("Calculate Estimate"):
            inputs = {
                "total_squares": total_squares,
                "total_area_sqft": total_area_sqft,
                "ridge_lf": ridge_lf,
                "hip_lf": hip_lf,
                "valley_lf": valley_lf,
                "eave_lf": eave_lf,
                "dripedge_lf": dripedge_lf,
            }
            line_items, subtotal, total = compute_estimate(inputs, pricing, waste_pct=float(waste_pct), op_pct=float(op_pct))

            st.subheader("Estimate")
            st.write("**Line items:**")
            if line_items:
                rows = []
                for name, item in line_items.items():
                    rows.append({
                        "Item": name,
                        "Qty": item["qty"],
                        "Unit": item["unit"],
                        "Rate": f"${item['rate']:.2f}",
                        "Cost": f"${item['cost']:.2f}",
                    })
                st.table(rows)
            else:
                st.info("No billable items detected.")

            st.markdown(f"### **Subtotal: ${subtotal:,.2f}**")
            st.markdown(f"### **Final Total (with O&P): ${total:,.2f}**")
