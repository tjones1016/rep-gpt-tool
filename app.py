import os
import re
import io
from typing import Dict, Any, Tuple

import streamlit as st
from dotenv import load_dotenv

# LangChain + RAG bits (unchanged, but now loads all files in /data)
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

# Set page config with logo as favicon/app icon
logo_path = os.path.join("data", "logo.png")
if os.path.exists(logo_path):
    st.set_page_config(
        page_title="Rep GPT — Chat + Estimates",
        page_icon=logo_path,   # uses your logo.png as favicon/home screen icon
        layout="wide"
    )
else:
    st.set_page_config(
        page_title="Rep GPT — Chat + Estimates",
        page_icon="📣",  # fallback emoji if logo is missing
        layout="wide"
    )

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
# Pricing guide parsing
# -----------------------------
@st.cache_data(show_spinner=False)
def read_pricing_text() -> str:
    """Read text from data/pricing_guide.docx if present."""
    pg_path = os.path.join("data", "pricing_guide.docx")
    if os.path.exists(pg_path):
        try:
            return docx2txt.process(pg_path) or ""
        except Exception:
            return ""
    return ""

PRICING_TEXT = read_pricing_text()

def _find_money(line: str) -> float | None:
    m = re.search(r"\$?\s*([0-9]+(?:\.[0-9]{1,2})?)", line.replace(",", ""))
    return float(m.group(1)) if m else None

def extract_pricing(pricing_text: str) -> Dict[str, float]:
    text = pricing_text.lower()
    pricing: Dict[str, float] = {}

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    def set_if_match(key: str, ln: str, unit_kw: str) -> None:
        if key not in pricing and unit_kw in ln:
            val = _find_money(ln)
            if val is not None:
                pricing[key] = val

    for ln in lines:
        if any(k in ln for k in ["shingle", "architectural", "asphalt"]) and "square" in ln:
            set_if_match("shingles_per_square", ln, "square")
        if "ridge" in ln and ("lf" in ln or "linear foot" in ln or "lineal foot" in ln):
            set_if_match("ridge_per_lf", ln, "lf")
        if "hip" in ln and ("lf" in ln or "linear foot" in ln or "lineal foot" in ln):
            set_if_match("hip_per_lf", ln, "lf")
        if "valley" in ln and ("lf" in ln or "linear foot" in ln or "lineal foot" in ln):
            set_if_match("valley_per_lf", ln, "lf")
        if any(k in ln for k in ["eave", "drip edge", "drip-edge", "starter"]):
            if ("lf" in ln or "linear foot" in ln or "lineal foot" in ln):
                set_if_match("eave_per_lf", ln, "lf")
        if "rake" in ln and ("lf" in ln or "linear foot" in ln or "lineal foot" in ln):
            set_if_match("rake_per_lf", ln, "lf")
        if "flashing" in ln and ("lf" in ln or "linear foot" in ln or "lineal foot" in ln):
            set_if_match("flashing_per_lf", ln, "lf")
        if "underlayment" in ln and "square" in ln:
            set_if_match("underlayment_per_square", ln, "square")
        if "labor" in ln and "square" in ln:
            set_if_match("labor_per_square", ln, "square")

    return pricing

DEFAULT_PRICING = extract_pricing(PRICING_TEXT)

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
        "rake_lf": 0.0,
        "flashing_lf": 0.0,
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
    grab_linear("rake", "rake_lf")
    grab_linear("flashing", "flashing_lf")

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
def compute_estimate(inputs: Dict[str, float], pricing: Dict[str, float], waste_pct: float = 10.0) -> Tuple[Dict[str, Any], float]:
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

    if "underlayment_per_square" in pricing:
        cost = adj_squares * p("underlayment_per_square")
        line_items["Underlayment"] = {"qty": round(adj_squares, 2), "unit": "sq", "rate": p("underlayment_per_square"), "cost": round(cost, 2)}

    if "labor_per_square" in pricing:
        cost = adj_squares * p("labor_per_square")
        line_items["Labor (per square)"] = {"qty": round(adj_squares, 2), "unit": "sq", "rate": p("labor_per_square"), "cost": round(cost, 2)}

    def add_linear(name: str, key_in: str, key_price: str):
        lf = float(inputs.get(key_in, 0.0))
        rate = p(key_price)
        if lf > 0 and rate > 0:
            line_items[name] = {"qty": lf, "unit": "lf", "rate": rate, "cost": round(lf * rate, 2)}

    add_linear("Ridge", "ridge_lf", "ridge_per_lf")
    add_linear("Hip", "hip_lf", "hip_per_lf")
    add_linear("Valley", "valley_lf", "valley_per_lf")
    add_linear("Eave/Drip Edge", "eave_lf", "eave_per_lf")
    add_linear("Rake", "rake_lf", "rake_per_lf")
    add_linear("Flashing", "flashing_lf", "flashing_per_lf")

    total = round(sum(item["cost"] for item in line_items.values()), 2)
    return line_items, total

# -----------------------------
# UI
# -----------------------------
tab_chat, tab_est = st.tabs(["💬 Chat", "📐 Estimate Calculator"])

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

with tab_est:
    st.subheader("Estimate Calculator (10% waste applied automatically)")
    st.markdown("**Pricing (auto-read from `data/pricing_guide.docx` when possible):**")
    with st.expander("Edit pricing", expanded=False):
        colA, colB, colC = st.columns(3)
        shingles_per_square = colA.number_input("Shingles ($/square)", value=float(DEFAULT_PRICING.get("shingles_per_square", 0.0)), min_value=0.0, step=1.0)
        labor_per_square = colB.number_input("Labor ($/square) [optional]", value=float(DEFAULT_PRICING.get("labor_per_square", 0.0)), min_value=0.0, step=1.0)
        underlayment_per_square = colC.number_input("Underlayment ($/square) [optional]", value=float(DEFAULT_PRICING.get("underlayment_per_square", 0.0)), min_value=0.0, step=1.0)

        ridge_per_lf = colA.number_input("Ridge ($/LF)", value=float(DEFAULT_PRICING.get("ridge_per_lf", 0.0)), min_value=0.0, step=0.5)
        hip_per_lf = colB.number_input("Hip ($/LF)", value=float(DEFAULT_PRICING.get("hip_per_lf", 0.0)), min_value=0.0, step=0.5)
        valley_per_lf = colC.number_input("Valley ($/LF)", value=float(DEFAULT_PRICING.get("valley_per_lf", 0.0)), min_value=0.0, step=0.5)

        eave_per_lf = colA.number_input("Eave/Drip Edge ($/LF)", value=float(DEFAULT_PRICING.get("eave_per_lf", 0.0)), min_value=0.0, step=0.5)
        rake_per_lf = colB.number_input("Rake ($/LF)", value=float(DEFAULT_PRICING.get("rake_per_lf", 0.0)), min_value=0.0, step=0.5)
        flashing_per_lf = colC.number_input("Flashing ($/LF)", value=float(DEFAULT_PRICING.get("flashing_per_lf", 0.0)), min_value=0.0, step=0.5)

    pricing = {
        "shingles_per_square": shingles_per_square,
        "labor_per_square": labor_per_square,
        "underlayment_per_square": underlayment_per_square,
        "ridge_per_lf": ridge_per_lf,
        "hip_per_lf": hip_per_lf,
        "valley_per_lf": valley_per_lf,
        "eave_per_lf": eave_per_lf,
        "rake_per_lf": rake_per_lf,
        "flashing_per_lf": flashing_per_lf,
    }

    st.markdown("**Input measurements (upload EagleView PDF or enter manually):**")
    col1, col2 = st.columns(2)

    with col1:
        ev_file = st.file_uploader("Upload EagleView PDF", type=["pdf"], help="We’ll extract area, squares, and linear feet if present.")
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
        eave_lf = st.number_input("Eave / Drip Edge (LF)", value=float(parsed_inputs.get("eave_lf", 0.0)), min_value=0.0, step=1.0)
        rake_lf = st.number_input("Rake (LF)", value=float(parsed_inputs.get("rake_lf", 0.0)), min_value=0.0, step=1.0)
        flashing_lf = st.number_input("Flashing (LF)", value=float(parsed_inputs.get("flashing_lf", 0.0)), min_value=0.0, step=1.0)

        waste_pct = st.slider("Waste % (default 10%)", min_value=0, max_value=25, value=10, step=1)

        if st.button("Calculate Estimate"):
            inputs = {
                "total_squares": total_squares,
                "total_area_sqft": total_area_sqft,
                "ridge_lf": ridge_lf,
                "hip_lf": hip_lf,
                "valley_lf": valley_lf,
                "eave_lf": eave_lf,
                "rake_lf": rake_lf,
                "flashing_lf": flashing_lf,
            }
            line_items, total = compute_estimate(inputs, pricing, waste_pct=float(waste_pct))

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
                st.info("No billable items detected. Add pricing and quantities to compute an
