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
# This ensures iPhones use it when users "Add to Home Screen"
st.set_page_config(
    page_title="Rep GPT — Chat + Estimates",
    page_icon="apple-touch-icon.png",  # your logo file
    layout="wide"
)

# Optional: enforce Apple Touch Icon in HTML (extra safety for iPhones)
st.markdown("""
<link rel="apple-touch-icon" sizes="180x180" href="apple-touch-icon.png">
""", unsafe_allow_html=True)

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
    """
    Heuristic parser for common line items.
    Looks for $ amounts with keywords and units like 'per square' or 'per lf/linear foot'.
    You can expand these keywords over time.
    """
    text = pricing_text.lower()
    pricing: Dict[str, float] = {}

    # Split into lines for heuristic matching
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Helper to set price if not already set and unit matches
    def set_if_match(key: str, ln: str, unit_kw: str) -> None:
        if key not in pricing and unit_kw in ln:
            val = _find_money(ln)
            if val is not None:
                pricing[key] = val

    for ln in lines:
        # Shingles per square
        if any(k in ln for k in ["shingle", "architectural", "asphalt"]) and "square" in ln:
            set_if_match("shingles_per_square", ln, "square")

        # Ridge per LF
        if "ridge" in ln and ("lf" in ln or "linear foot" in ln or "lineal foot" in ln):
            set_if_match("ridge_per_lf", ln, "lf")

        # Hip per LF
        if "hip" in ln and ("lf" in ln or "linear foot" in ln or "lineal foot" in ln):
            set_if_match("hip_per_lf", ln, "lf")

        # Valley per LF
        if "valley" in ln and ("lf" in ln or "linear foot" in ln or "lineal foot" in ln):
            set_if_match("valley_per_lf", ln, "lf")

        # Eave / Drip edge per LF
        if any(k in ln for k in ["eave", "drip edge", "drip-edge", "starter"]):
            if ("lf" in ln or "linear foot" in ln or "lineal foot" in ln):
                set_if_match("eave_per_lf", ln, "lf")

        # Rake per LF
        if "rake" in ln and ("lf" in ln or "linear foot" in ln or "lineal foot" in ln):
            set_if_match("rake_per_lf", ln, "lf")

        # Flashing per LF
        if "flashing" in ln and ("lf" in ln or "linear foot" in ln or "lineal foot" in ln):
            set_if_match("flashing_per_lf", ln, "lf")

        # Underlayment per square
        if "underlayment" in ln and "square" in ln:
            set_if_match("underlayment_per_square", ln, "square")

        # Labor per square (optional)
        if "labor" in ln and "square" in ln:
            set_if_match("labor_per_square", ln, "square")

    return pricing

DEFAULT_PRICING = extract_pricing(PRICING_TEXT)

# -----------------------------
# EagleView parsing
# -----------------------------
def parse_eagleview_text(text: str) -> Dict[str, float]:
    """
    Heuristic extraction from EagleView text.
    Pulls:
      - total_squares (or total area sq ft)
      - ridge_lf, hip_lf, valley_lf, eave_lf, rake_lf, flashing_lf (if present)
    Always returns floats (0.0 if missing).
    """
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

    # squares
    m_sq = re.search(r"total squares\D*([0-9]+(?:\.[0-9]+)?)", t)
    if m_sq:
        data["total_squares"] = float(m_sq.group(1))

    # area in sqft
    m_area = re.search(r"(total (?:roof )?area|roof area)\D*([0-9]+(?:\.[0-9]+)?)\s*(sq\s*ft|square feet|sf)", t)
    if m_area:
        data["total_area_sqft"] = float(m_area.group(2))

    # linears
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
# NEW ESTIMATOR: compute_estimate_v2
# -----------------------------
def compute_estimate_v2(inputs: Dict[str, float],
                        settings: Dict[str, float],
                        waste_pct: float = 10.0,
                        apply_op: bool = False) -> Tuple[Dict[str, Any], float]:
    """
    Returns (line_items, total_cost)
    - inputs: eagleview/extracted inputs (squares, area, linears)
    - settings: default rates (shingle bundled, tearoff, ridge, ridgevent, eave, pipejack, turtle, exhaust, dumpster)
    - waste_pct: percent waste to apply to squares
    - apply_op: whether to apply 20% O&P to subtotal
    """
    # Decide base squares
    squares = inputs.get("total_squares", 0.0)
    if squares <= 0:
        sqft = inputs.get("total_area_sqft", 0.0)
        if sqft > 0:
            squares = sqft / 100.0

    # Apply waste to squares
    adj_squares = squares * (1.0 + waste_pct / 100.0)

    # Helper to read settings with fallback 0
    def s(key: str) -> float:
        return float(settings.get(key, 0.0))

    line_items: Dict[str, Any] = {}

    # Shingles (bundled includes labor & underlayment)
    shingle_rate = s("shingles_bundle_large") if squares >= s("large_roof_threshold") else s("shingles_bundle_small")
    shingle_cost = adj_squares * shingle_rate
    line_items["Shingles (bundled)"] = {"qty": round(adj_squares, 2), "unit": "sq", "rate": round(shingle_rate, 2), "cost": round(shingle_cost, 2)}

    # Tear-off & disposal per square
    tearoff_cost = squares * s("tearoff_per_sq")
    line_items["Tear-off & disposal"] = {"qty": round(squares, 2), "unit": "sq", "rate": round(s("tearoff_per_sq"), 2), "cost": round(tearoff_cost, 2)}

    # Ridge cap shingles (per LF)
    ridge_lf = float(inputs.get("ridge_lf", 0.0))
    if ridge_lf > 0:
        rc_cost = ridge_lf * s("ridge_cap_per_lf")
        line_items["Ridge cap shingles"] = {"qty": ridge_lf, "unit": "lf", "rate": round(s("ridge_cap_per_lf"), 2), "cost": round(rc_cost, 2)}

    # Ridge vent (per LF)
    ridgevent_lf = float(inputs.get("ridgevent_lf", inputs.get("ridge_lf", 0.0)))
    if ridgevent_lf > 0 and s("ridgevent_per_lf") > 0:
        rv_cost = ridgevent_lf * s("ridgevent_per_lf")
        line_items["Ridge vent"] = {"qty": ridgevent_lf, "unit": "lf", "rate": round(s("ridgevent_per_lf"), 2), "cost": round(rv_cost, 2)}

    # Linear items: hip / valley / eave / rake / flashing
    def add_linear(name: str, input_key: str, setting_key: str):
        lf = float(inputs.get(input_key, 0.0))
        rate = s(setting_key)
        if lf > 0 and rate > 0:
            line_items[name] = {"qty": lf, "unit": "lf", "rate": round(rate, 2), "cost": round(lf * rate, 2)}

    add_linear("Hip", "hip_lf", "hip_per_lf")
    add_linear("Valley", "valley_lf", "valley_per_lf")
    add_linear("Eave / Drip Edge", "eave_lf", "eave_per_lf")
    add_linear("Rake", "rake_lf", "rake_per_lf")
    add_linear("Flashing", "flashing_lf", "flashing_per_lf")

    # Penetrations & accessories (user-entered quantities in UI will map to these keys)
    pipe_jacks = float(inputs.get("pipe_jacks", 0.0))
    if pipe_jacks > 0 and s("pipejack_per_each") > 0:
        line_items["Pipe jacks"] = {"qty": int(pipe_jacks), "unit": "ea", "rate": round(s("pipejack_per_each"), 2), "cost": round(pipe_jacks * s("pipejack_per_each"), 2)}

    turtle_vents = float(inputs.get("turtle_vents", 0.0))
    if turtle_vents > 0 and s("turtle_per_each") > 0:
        line_items["Turtle vents"] = {"qty": int(turtle_vents), "unit": "ea", "rate": round(s("turtle_per_each"), 2), "cost": round(turtle_vents * s("turtle_per_each"), 2)}

    exhaust_caps = float(inputs.get("exhaust_caps", 0.0))
    if exhaust_caps > 0 and s("exhaust_per_each") > 0:
        line_items["Exhaust caps"] = {"qty": int(exhaust_caps), "unit": "ea", "rate": round(s("exhaust_per_each"), 2), "cost": round(exhaust_caps * s("exhaust_per_each"), 2)}

    # Dumpster / haul-off (flat)
    dumpster_cost = s("dumpster_flat")
    if dumpster_cost > 0:
        line_items["Dumpster / Haul-off"] = {"qty": 1, "unit": "ea", "rate": round(dumpster_cost, 2), "cost": round(dumpster_cost, 2)}

    # Subtotal
    subtotal = round(sum(item["cost"] for item in line_items.values()), 2)

    # Apply O&P if requested
    if apply_op:
        op_amount = round(subtotal * 0.20, 2)
        line_items["Overhead & Profit (20%)"] = {"qty": 1, "unit": "ea", "rate": "20%", "cost": op_amount}
        subtotal = round(subtotal + op_amount, 2)

    return line_items, subtotal

# -----------------------------
# UI (Tabbed layout: Chat + Estimate)
# -----------------------------
tab_chat, tab_est = st.tabs(["💬 Chat", "📐 Estimate Calculator"])

# ----- Chat tab -----
with tab_chat:
    st.caption("Ask about sales, pay, objections, pricing, or process steps. Powered by your `/data` docs.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Type your question…"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        result = qa_chain.run({"question": prompt, "chat_history": st.session_state.chat_history})
        st.chat_message("assistant").markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.session_state.chat_history.append((prompt, result))

# ----- Estimate tab -----
with tab_est:
    st.subheader("Estimate Calculator (waste adjustable)")

    # 1) Default settings / rates (read from pricing_guide if available)
    st.markdown("**Default Rates (auto-read from `data/pricing_guide.docx` when possible):**")
    with st.expander("Edit default rates", expanded=False):
        col1, col2, col3 = st.columns(3)

        # Shingle bundled rates (your request: $435 large, $450 small)
        col1.markdown("**Shingle bundle (includes labor & underlayment)**")
        shingles_bundle_large = col1.number_input("Shingles bundle - Large roofs ($/sq)", value=float(DEFAULT_PRICING.get("shingles_bundle_large", 435.0)), min_value=0.0, step=1.0)
        shingles_bundle_small = col1.number_input("Shingles bundle - Small roofs ($/sq)", value=float(DEFAULT_PRICING.get("shingles_bundle_small", 450.0)), min_value=0.0, step=1.0)
        large_roof_threshold = col1.number_input("Large roof threshold (sq)", value=float(DEFAULT_PRICING.get("large_roof_threshold", 30.0)), min_value=1.0, step=1.0)

        # Tear-off & disposal
        col2.markdown("**Removal & Disposal**")
        tearoff_per_sq = col2.number_input("Tear-off & disposal ($/sq)", value=float(DEFAULT_PRICING.get("tearoff_per_sq", 65.0)), min_value=0.0, step=1.0)
        dumpster_flat = col2.number_input("Dumpster / Haul-off (flat)", value=float(DEFAULT_PRICING.get("dumpster_flat", 600.0)), min_value=0.0, step=10.0)

        # Ridge / ridge vent / eave / linear rates
        col3.markdown("**Linear & accessory rates**")
        ridge_cap_per_lf = col3.number_input("Ridge cap shingles ($/lf)", value=float(DEFAULT_PRICING.get("ridge_per_lf", 5.0)), min_value=0.0, step=0.5)
        ridgevent_per_lf = col3.number_input("Ridge vent ($/lf)", value=float(DEFAULT_PRICING.get("ridgevent_per_lf", 11.0)), min_value=0.0, step=0.5)
        eave_per_lf = col3.number_input("Eave / Drip edge ($/lf)", value=float(DEFAULT_PRICING.get("eave_per_lf", 1.5)), min_value=0.0, step=0.25)

        # Other linears
        hip_per_lf = col1.number_input("Hip ($/lf)", value=float(DEFAULT_PRICING.get("hip_per_lf", 0.0)), min_value=0.0, step=0.5)
        valley_per_lf = col2.number_input("Valley ($/lf)", value=float(DEFAULT_PRICING.get("valley_per_lf", 0.0)), min_value=0.0, step=0.5)
        rake_per_lf = col3.number_input("Rake ($/lf)", value=float(DEFAULT_PRICING.get("rake_per_lf", 0.0)), min_value=0.0, step=0.5)
        flashing_per_lf = col1.number_input("Flashing ($/lf)", value=float(DEFAULT_PRICING.get("flashing_per_lf", 0.0)), min_value=0.0, step=0.5)

        # Penetrations / accessories per each
        pipejack_per_each = col2.number_input("Pipe jack ($/each)", value=float(DEFAULT_PRICING.get("pipejack_per_each", 45.0)), min_value=0.0, step=1.0)
        turtle_per_each = col3.number_input("Turtle vent ($/each)", value=float(DEFAULT_PRICING.get("turtle_per_each", 65.0)), min_value=0.0, step=1.0)
        exhaust_per_each = col1.number_input("Exhaust cap ($/each)", value=float(DEFAULT_PRICING.get("exhaust_per_each", 100.0)), min_value=0.0, step=1.0)

    # Bundle settings into a dict for compute_estimate_v2
    settings = {
        "shingles_bundle_large": shingles_bundle_large,
        "shingles_bundle_small": shingles_bundle_small,
        "large_roof_threshold": large_roof_threshold,
        "tearoff_per_sq": tearoff_per_sq,
        "dumpster_flat": dumpster_flat,
        "ridge_cap_per_lf": ridge_cap_per_lf,
        "ridgevent_per_lf": ridgevent_per_lf,
        "eave_per_lf": eave_per_lf,
        "hip_per_lf": hip_per_lf,
        "valley_per_lf": valley_per_lf,
        "rake_per_lf": rake_per_lf,
        "flashing_per_lf": flashing_per_lf,
        "pipejack_per_each": pipejack_per_each,
        "turtle_per_each": turtle_per_each,
        "exhaust_per_each": exhaust_per_each,
    }

    st.markdown("**Input measurements (upload EagleView PDF or enter manually):**")
    colA, colB = st.columns(2)

    # Upload EagleView PDF
    with colA:
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

    # Manual entry (pre-filled from parsed data if available)
    with colB:
        st.caption("Manual override / entry")
        total_squares = st.number_input("Total Squares", value=float(parsed_inputs.get("total_squares", 0.0)), min_value=0.0, step=0.1)
        total_area_sqft = st.number_input("Total Area (sq ft)", value=float(parsed_inputs.get("total_area_sqft", 0.0)), min_value=0.0, step=10.0)
        ridge_lf = st.number_input("Ridge (LF)", value=float(parsed_inputs.get("ridge_lf", 0.0)), min_value=0.0, step=1.0)
        hip_lf = st.number_input("Hip (LF)", value=float(parsed_inputs.get("hip_lf", 0.0)), min_value=0.0, step=1.0)
        valley_lf = st.number_input("Valley (LF)", value=float(parsed_inputs.get("valley_lf", 0.0)), min_value=0.0, step=1.0)
        eave_lf = st.number_input("Eave / Drip Edge (LF)", value=float(parsed_inputs.get("eave_lf", 0.0)), min_value=0.0, step=1.0)
        rake_lf = st.number_input("Rake (LF)", value=float(parsed_inputs.get("rake_lf", 0.0)), min_value=0.0, step=1.0)
        flashing_lf = st.number_input("Flashing (LF)", value=float(parsed_inputs.get("flashing_lf", 0.0)), min_value=0.0, step=1.0)

        # Penetrations (user-entered)
        pipe_jacks = st.number_input("Pipe Jacks (each)", min_value=0, value=0, step=1)
        turtle_vents = st.number_input("Turtle Vents (each)", min_value=0, value=0, step=1)
        exhaust_caps = st.number_input("Exhaust Caps (each)", min_value=0, value=0, step=1)

        waste_pct = st.slider("Waste % (default 10%)", min_value=0, max_value=25, value=10, step=1)
        apply_op = st.checkbox("Apply Overhead & Profit (20%)", value=False)

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
                "pipe_jacks": pipe_jacks,
                "turtle_vents": turtle_vents,
                "exhaust_caps": exhaust_caps,
                # optional: let users input a separate ridgevent_lf if EagleView provides it
                "ridgevent_lf": float(parsed_inputs.get("ridge_lf", 0.0))
            }

            line_items, total = compute_estimate_v2(inputs, settings, waste_pct=float(waste_pct), apply_op=apply_op)

            st.subheader("Estimate")
            if line_items:
                # Build a displayable table
                rows = []
                for name, item in line_items.items():
                    # item may have qty/unit/rate/cost
                    qty = item.get("qty", "")
                    unit = item.get("unit", "")
                    rate = item.get("rate", "")
                    cost = item.get("cost", 0.0)
                    rows.append({
                        "Item": name,
                        "Qty": qty,
                        "Unit": unit,
                        "Rate": f"{rate}" if isinstance(rate, str) else f"${rate:.2f}",
                        "Cost": f"${cost:,.2f}"
                    })
                st.table(rows)
            else:
                st.info("No billable items detected. Add pricing and quantities to compute an estimate.")

            st.markdown(f"### **Total: ${total:,.2f}**")
            st.caption("Note: Squares include the selected waste percentage.")

# End of file
