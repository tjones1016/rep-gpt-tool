import os
import re
import io
from typing import Dict, Any, Tuple

import streamlit as st
from dotenv import load_dotenv

# -----------------------------
# Page setup + iOS icon
# -----------------------------
st.set_page_config(
    page_title="Rep GPT — Chat + Estimates",
    page_icon="apple-touch-icon.png",
    layout="wide",
)

st.markdown(
    """
<link rel="apple-touch-icon" sizes="180x180" href="apple-touch-icon.png">
""",
    unsafe_allow_html=True,
)

# -----------------------------
# LangChain / RAG
# -----------------------------
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
            allow_dangerous_deserialization=True,
        )
    else:
        db = FAISS.from_documents(split_docs, OpenAIEmbeddings())
        db.save_local("vectorstore")

    retriever = db.as_retriever()
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    return ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

qa_chain = setup_conversational_chain()

# -----------------------------
# Pricing guide parsing (lightweight heuristics)
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
        if any(k in ln for k in ["architectural", "shingle", "asphalt"]) and "square" in ln:
            set_if_match("shingles_per_square", ln, "square")
        if "ridge" in ln and ("lf" in ln or "linear foot" in ln or "lineal foot" in ln):
            set_if_match("ridge_per_lf", ln, "lf")
        if "hip" in ln and ("lf" in ln or "linear foot" in ln or "lineal foot" in ln):
            set_if_match("hip_per_lf", ln, "lf")
        if "valley" in ln and ("lf" in ln or "linear foot" in ln or "lineal foot" in ln):
            set_if_match("valley_per_lf", ln, "lf")
        if any(k in ln for k in ["drip edge", "eave", "drip-edge", "starter"]):
            if ("lf" in ln or "linear foot" in ln or "lineal foot" in ln):
                set_if_match("drip_edge_per_lf", ln, "lf")
        if "underlayment" in ln and "square" in ln:
            set_if_match("underlayment_per_square", ln, "square")
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

    for lbl, key in [
        ("ridge", "ridge_lf"),
        ("hip", "hip_lf"),
        ("valley", "valley_lf"),
        ("eave", "eave_lf"),
        ("rake", "rake_lf"),
        ("flashing", "flashing_lf"),
    ]:
        grab_linear(lbl, key)

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
# Estimate math (expanded)
# -----------------------------

def compute_estimate(
    inputs: Dict[str, float],
    pricing: Dict[str, float],
    waste_pct: float = 10.0,
    roofing_system: str = "architectural",  # 'architectural' | 'standing_seam' | 'exposed_fastener' | 'fortified'
    use_tiered_arch: bool = False,
    arch_small_rate: float = 450.0,
    arch_large_rate: float = 435.0,
    tier_break_sq: float = 20.0,
    ridge_vent_mode: str = "add",  # 'add' or 'cut'
    insurance_mode: bool = False,
    op_overhead_pct: float = 10.0,
    op_profit_pct: float = 10.0,
    deductible: float = 0.0,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Returns (line_items, summary)
    - line_items: dict[name] -> {qty, unit, rate, cost}
    - summary: {subtotal, op_overhead, op_profit, total_before_deductible, deductible, total_due}
    """
    def p(key: str, default: float = 0.0) -> float:
        return float(pricing.get(key, default))

    # Determine squares
    squares = inputs.get("total_squares", 0.0)
    if squares <= 0:
        sqft = inputs.get("total_area_sqft", 0.0)
        if sqft > 0:
            squares = sqft / 100.0

    # Apply waste to *installation* quantities, not tear-off
    inst_squares = squares * (1.0 + waste_pct / 100.0)

    line_items: Dict[str, Any] = {}

    # ---------------- Base roofing per-square by system ----------------
    rate_label = "Shingles"
    base_rate = p("shingles_per_square")

    if roofing_system == "architectural":
        if use_tiered_arch:
            base_rate = arch_small_rate if squares < tier_break_sq else arch_large_rate
            rate_label = "Architectural Shingles (tiered)"
        else:
            rate_label = "Architectural Shingles"
            if base_rate <= 0:
                base_rate = arch_large_rate  # sensible default
    elif roofing_system == "standing_seam":
        base_rate = p("standing_seam_per_square", 1250.0)
        rate_label = "Standing Seam Metal"
    elif roofing_system == "exposed_fastener":
        base_rate = p("exposed_fastener_per_square", 900.0)
        rate_label = "Exposed-Fastener Metal"
    elif roofing_system == "fortified":
        base_rate = p("fortified_per_square", 600.0)
        rate_label = "FORTIFIED Roof"

    if base_rate > 0 and inst_squares > 0:
        cost = inst_squares * base_rate
        line_items[f"{rate_label} (with waste)"] = {
            "qty": round(inst_squares, 2),
            "unit": "sq",
            "rate": round(base_rate, 2),
            "cost": round(cost, 2),
        }

    # ---------------- Tear-off per layer (no waste) ----------------
    layers = max(1.0, float(inputs.get("layers", 1.0))) if p("tearoff_per_square", 0.0) > 0 else float(inputs.get("layers", 0.0))
    if layers and p("tearoff_per_square", 0.0) > 0:
        cost = squares * layers * p("tearoff_per_square")
        line_items["Tear-off (per layer)"] = {
            "qty": round(squares * layers, 2),
            "unit": "sq",
            "rate": p("tearoff_per_square"),
            "cost": round(cost, 2),
        }

    # ---------------- Underlayment (per sq, uses install squares) ----------------
    if p("underlayment_per_square", 0.0) > 0 and inst_squares > 0:
        cost = inst_squares * p("underlayment_per_square")
        line_items["Underlayment"] = {
            "qty": round(inst_squares, 2),
            "unit": "sq",
            "rate": p("underlayment_per_square"),
            "cost": round(cost, 2),
        }

    # ---------------- Linear items ----------------
    def add_linear(name: str, key_in: str, price_key: str):
        lf = float(inputs.get(key_in, 0.0))
        rate = p(price_key, 0.0)
        if lf > 0 and rate > 0:
            line_items[name] = {"qty": round(lf, 2), "unit": "lf", "rate": round(rate, 2), "cost": round(lf * rate, 2)}

    add_linear("Ridge", "ridge_lf", "ridge_per_lf")
    add_linear("Hip", "hip_lf", "hip_per_lf")
    add_linear("Valley", "valley_lf", "valley_per_lf")
    add_linear("Drip Edge", "eave_lf", "drip_edge_per_lf")
    add_linear("Rake", "rake_lf", "rake_per_lf")
    add_linear("Flashing", "flashing_lf", "flashing_per_lf")

    # Ridge vent pricing choice (Add vs Cut)
    ridge_vent_lf = float(inputs.get("ridge_vent_lf", 0.0))
    if ridge_vent_lf > 0:
        rv_rate = p("ridge_vent_add_per_lf", 5.0) if ridge_vent_mode == "add" else p("ridge_vent_cut_per_lf", 4.0)
        line_items[f"Ridge Vent ({'Add' if ridge_vent_mode=='add' else 'Cut'})"] = {
            "qty": round(ridge_vent_lf, 2),
            "unit": "lf",
            "rate": round(rv_rate, 2),
            "cost": round(ridge_vent_lf * rv_rate, 2),
        }

    # ---------------- Per-each accessories ----------------
    def add_each(name: str, count_key: str, price_key: str):
        n = float(inputs.get(count_key, 0.0))
        rate = p(price_key, 0.0)
        if n > 0 and rate > 0:
            line_items[name] = {"qty": int(n), "unit": "ea", "rate": round(rate, 2), "cost": round(n * rate, 2)}

    add_each("Pipe Boots", "pipe_boots", "pipe_boot_each")
    add_each("Roof Vents (turtle)", "roof_vents", "roof_vent_each")
    add_each("Exhaust Caps", "exhaust_caps", "exhaust_cap_each")

    # Attic fans: price with minimum job
    attic_fans = float(inputs.get("attic_fans", 0.0))
    if attic_fans > 0 and (p("attic_fan_each", 0.0) > 0 or p("attic_fan_min_job", 0.0) > 0):
        per_cost = attic_fans * p("attic_fan_each", 300.0)
        cost = max(per_cost, p("attic_fan_min_job", 500.0))
        line_items["Attic Fans"] = {"qty": int(attic_fans), "unit": "ea", "rate": round(p("attic_fan_each", 300.0), 2), "cost": round(cost, 2)}

    # Chimney removals (include plywood material per sheet)
    plywood_rate = p("plywood_per_sheet", 45.0)
    for label, key, price_key in [
        ("Chimney Removal (Small)", "chimney_small", "chimney_small_each"),
        ("Chimney Removal (Large)", "chimney_large", "chimney_large_each"),
    ]:
        n = float(inputs.get(key, 0.0))
        base_rate = p(price_key, 0.0)
        if n > 0 and base_rate > 0:
            cost = n * (base_rate + plywood_rate)
            line_items[label] = {"qty": int(n), "unit": "ea", "rate": round(base_rate + plywood_rate, 2), "cost": round(cost, 2)}

    # Skylight R&R: labor + materials + margin
    skylights = float(inputs.get("skylights", 0.0))
    if skylights > 0 and (p("skylight_labor_each", 350.0) > 0 or p("skylight_material_each", 0.0) > 0):
        labor = skylights * p("skylight_labor_each", 350.0)
        material = skylights * p("skylight_material_each", 0.0)
        margin = p("material_margin_pct", 0.0) / 100.0
        cost = labor + material * (1 + margin)
        line_items["Skylight R&R"] = {"qty": int(skylights), "unit": "ea", "rate": round((labor + material)/max(skylights,1), 2), "cost": round(cost, 2)}

    # ---------------- Flat fees ----------------
    def add_flat(name: str, price_key: str, cond: bool = True):
        rate = p(price_key, 0.0)
        if cond and rate > 0:
            line_items[name] = {"qty": 1, "unit": "flat", "rate": round(rate, 2), "cost": round(rate, 2)}

    add_flat("Dumpster", "dumpster_flat")
    add_flat("Permit Fee", "permit_flat")
    add_flat("Mobilization", "mobilization_flat")

    # ---------------- Totals + Insurance logic ----------------
    subtotal = round(sum(item["cost"] for item in line_items.values()), 2)

    op_overhead = round(subtotal * (op_overhead_pct / 100.0), 2) if insurance_mode else 0.0
    op_profit = round((subtotal + op_overhead) * (op_profit_pct / 100.0), 2) if insurance_mode else 0.0

    total_before_deductible = round(subtotal + op_overhead + op_profit, 2)
    total_due = round(max(total_before_deductible - float(deductible), 0.0), 2) if insurance_mode else total_before_deductible

    summary = {
        "subtotal": subtotal,
        "op_overhead": op_overhead,
        "op_profit": op_profit,
        "total_before_deductible": total_before_deductible,
        "deductible": float(deductible) if insurance_mode else 0.0,
        "total_due": total_due,
    }

    return line_items, summary

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
    st.subheader("Estimate Calculator (waste applies to install items)")

    # ===================== Pricing =====================
    st.markdown("**Pricing (auto-read from `data/pricing_guide.docx` when possible; you can override below):**")
    with st.expander("Edit pricing", expanded=False):
        colA, colB, colC = st.columns(3)

        # Base systems
        standing_seam_per_square = colA.number_input("Standing Seam Metal ($/SQ)", min_value=0.0, value=float(DEFAULT_PRICING.get("standing_seam_per_square", 1250.0)), step=10.0)
        exposed_fastener_per_square = colB.number_input("Exposed-Fastener Metal ($/SQ)", min_value=0.0, value=float(DEFAULT_PRICING.get("exposed_fastener_per_square", 900.0)), step=10.0)
        fortified_per_square = colC.number_input("FORTIFIED ($/SQ)", min_value=0.0, value=float(DEFAULT_PRICING.get("fortified_per_square", 600.0)), step=10.0)

        # Architectural shingles
        st.markdown("**Architectural shingles** (tiered option available)")
        col1, col2, col3, col4 = st.columns(4)
        shingles_per_square = col1.number_input("Architectural ($/SQ) [single rate]", min_value=0.0, value=float(DEFAULT_PRICING.get("shingles_per_square", 435.0)), step=5.0)
        use_tiered_arch = col2.checkbox("Use tiered pricing (<20 SQ vs ≥20 SQ)", value=True)
        arch_small_rate = col3.number_input("<20 SQ rate", min_value=0.0, value=450.0, step=5.0)
        arch_large_rate = col4.number_input("≥20 SQ rate", min_value=0.0, value=435.0, step=5.0)
        tier_break_sq = col2.number_input("Tier breakpoint (SQ)", min_value=0.0, value=20.0, step=1.0)

        # Per-square adds
        underlayment_per_square = colA.number_input("Underlayment ($/SQ)", min_value=0.0, value=float(DEFAULT_PRICING.get("underlayment_per_square", 0.0)), step=1.0)
        tearoff_per_square = colB.number_input("Tear-off ($/SQ per layer)", min_value=0.0, value=45.0, step=1.0)
        labor_per_square = colC.number_input("Labor ($/SQ) [optional]", min_value=0.0, value=float(DEFAULT_PRICING.get("labor_per_square", 0.0)), step=1.0)

        # Linear
        ridge_per_lf = colA.number_input("Ridge ($/LF)", min_value=0.0, value=float(DEFAULT_PRICING.get("ridge_per_lf", 0.0)), step=0.5)
        hip_per_lf = colB.number_input("Hip ($/LF)", min_value=0.0, value=float(DEFAULT_PRICING.get("hip_per_lf", 0.0)), step=0.5)
        valley_per_lf = colC.number_input("Valley ($/LF)", min_value=0.0, value=float(DEFAULT_PRICING.get("valley_per_lf", 0.0)), step=0.5)
        drip_edge_per_lf = colA.number_input("Drip Edge ($/LF)", min_value=0.0, value=float(DEFAULT_PRICING.get("drip_edge_per_lf", 1.5)), step=0.5)
        rake_per_lf = colB.number_input("Rake ($/LF)", min_value=0.0, value=float(DEFAULT_PRICING.get("rake_per_lf", 0.0)), step=0.5)
        flashing_per_lf = colC.number_input("Flashing ($/LF)", min_value=0.0, value=float(DEFAULT_PRICING.get("flashing_per_lf", 0.0)), step=0.5)

        # Ridge vent options
        st.markdown("**Ridge Vent**")
        colrv1, colrv2 = st.columns(2)
        ridge_vent_add_per_lf = colrv1.number_input("Add RV ($/LF)", min_value=0.0, value=5.0, step=0.5)
        ridge_vent_cut_per_lf = colrv2.number_input("Cut RV ($/LF)", min_value=0.0, value=4.0, step=0.5)

        # Each items
        st.markdown("**Per-each items**")
        colE1, colE2, colE3, colE4 = st.columns(4)
        pipe_boot_each = colE1.number_input("Pipe Boot ($/ea)", min_value=0.0, value=50.0, step=5.0)
        roof_vent_each = colE2.number_input("Roof Vent / Turtle ($/ea)", min_value=0.0, value=65.0, step=5.0)
        exhaust_cap_each = colE3.number_input("Exhaust Cap ($/ea)", min_value=0.0, value=100.0, step=5.0)
        attic_fan_each = colE4.number_input("Attic Fan ($/ea)", min_value=0.0, value=300.0, step=10.0)
        attic_fan_min_job = colE4.number_input("Attic Fan Min Job ($)", min_value=0.0, value=500.0, step=10.0)

        # Chimneys & Skylights
        st.markdown("**Chimneys & Skylights**")
        colC1, colC2, colC3, colC4 = st.columns(4)
        chimney_small_each = colC1.number_input("Small Chimney Removal ($/ea) [labor]", min_value=0.0, value=250.0, step=10.0)
        chimney_large_each = colC2.number_input("Large Chimney Removal ($/ea) [labor]", min_value=0.0, value=500.0, step=10.0)
        plywood_per_sheet = colC3.number_input("Plywood ($/sheet)", min_value=0.0, value=45.0, step=1.0)
        skylight_labor_each = colC4.number_input("Skylight R&R Labor ($/ea)", min_value=0.0, value=350.0, step=10.0)
        skylight_material_each = colC4.number_input("Skylight Material ($/ea)", min_value=0.0, value=0.0, step=10.0)
        material_margin_pct = colC4.number_input("Material Margin (%)", min_value=0.0, value=0.0, step=1.0)

        # Flat fees
        st.markdown("**Flat fees**")
        colF1, colF2, colF3 = st.columns(3)
        dumpster_flat = colF1.number_input("Dumpster ($ flat)", min_value=0.0, value=600.0, step=25.0)
        permit_flat = colF2.number_input("Permit ($ flat)", min_value=0.0, value=0.0, step=25.0)
        mobilization_flat = colF3.number_input("Mobilization ($ flat)", min_value=0.0, value=0.0, step=25.0)

    # Assemble pricing dict for engine
    pricing = {
        # Base systems
        "shingles_per_square": shingles_per_square,
        "standing_seam_per_square": standing_seam_per_square,
        "exposed_fastener_per_square": exposed_fastener_per_square,
        "fortified_per_square": fortified_per_square,

        # Per-square
        "underlayment_per_square": underlayment_per_square,
        "tearoff_per_square": tearoff_per_square,
        "labor_per_square": labor_per_square,

        # Linear
        "ridge_per_lf": ridge_per_lf,
        "hip_per_lf": hip_per_lf,
        "valley_per_lf": valley_per_lf,
        "drip_edge_per_lf": drip_edge_per_lf,
        "rake_per_lf": rake_per_lf,
        "flashing_per_lf": flashing_per_lf,

        # Ridge vent
        "ridge_vent_add_per_lf": ridge_vent_add_per_lf,
        "ridge_vent_cut_per_lf": ridge_vent_cut_per_lf,

        # Each items
        "pipe_boot_each": pipe_boot_each,
        "roof_vent_each": roof_vent_each,
        "exhaust_cap_each": exhaust_cap_each,
        "attic_fan_each": attic_fan_each,
        "attic_fan_min_job": attic_fan_min_job,

        # Chimneys & Skylights
        "chimney_small_each": chimney_small_each,
        "chimney_large_each": chimney_large_each,
        "plywood_per_sheet": plywood_per_sheet,
        "skylight_labor_each": skylight_labor_each,
        "skylight_material_each": skylight_material_each,
        "material_margin_pct": material_margin_pct,

        # Flats
        "dumpster_flat": dumpster_flat,
        "permit_flat": permit_flat,
        "mobilization_flat": mobilization_flat,
    }

    # ===================== Measurements =====================
    st.markdown("**Input measurements (upload EagleView PDF or enter manually):**")
    col1, col2 = st.columns(2)

    # Upload EagleView
    parsed_inputs = {}
    with col1:
        ev_file = st.file_uploader("Upload EagleView PDF", type=["pdf"], help="We’ll extract area, squares, and linear feet if present.")
        if ev_file is not None:
            try:
                pdf_text = read_pdf_text(ev_file.read())
                parsed_inputs = parse_eagleview_text(pdf_text)
                st.success("Parsed EagleView values from PDF.")
                st.json({k: round(v, 2) for k, v in parsed_inputs.items() if v})
            except Exception as e:
                st.error(f"Could not read PDF: {e}")

    # Manual entry
    with col2:
        st.caption("Manual override / entry")
        total_squares = st.number_input("Total Squares", value=float(parsed_inputs.get("total_squares", 0.0)), min_value=0.0, step=0.1)
        total_area_sqft = st.number_input("Total Area (sq ft)", value=float(parsed_inputs.get("total_area_sqft", 0.0)), min_value=0.0, step=10.0)
        layers = st.number_input("Tear-off Layers", value=1.0, min_value=0.0, step=1.0)
        ridge_lf = st.number_input("Ridge (LF)", value=float(parsed_inputs.get("ridge_lf", 0.0)), min_value=0.0, step=1.0)
        hip_lf = st.number_input("Hip (LF)", value=float(parsed_inputs.get("hip_lf", 0.0)), min_value=0.0, step=1.0)
        valley_lf = st.number_input("Valley (LF)", value=float(parsed_inputs.get("valley_lf", 0.0)), min_value=0.0, step=1.0)
        eave_lf = st.number_input("Eave / Drip Edge (LF)", value=float(parsed_inputs.get("eave_lf", 0.0)), min_value=0.0, step=1.0)
        rake_lf = st.number_input("Rake (LF)", value=float(parsed_inputs.get("rake_lf", 0.0)), min_value=0.0, step=1.0)
        flashing_lf = st.number_input("Flashing (LF)", value=float(parsed_inputs.get("flashing_lf", 0.0)), min_value=0.0, step=1.0)
        ridge_vent_lf = st.number_input("Ridge Vent (LF)", value=0.0, min_value=0.0, step=1.0)

        # Counts
        st.markdown("**Counts**")
        colCnt1, colCnt2, colCnt3, colCnt4 = st.columns(4)
        pipe_boots = colCnt1.number_input("Pipe Boots (ea)", min_value=0, value=0, step=1)
        roof_vents = colCnt2.number_input("Roof Vents (ea)", min_value=0, value=0, step=1)
        exhaust_caps = colCnt3.number_input("Exhaust Caps (ea)", min_value=0, value=0, step=1)
        attic_fans = colCnt4.number_input("Attic Fans (ea)", min_value=0, value=0, step=1)

        st.markdown("**Chimneys & Skylights**")
        colCS1, colCS2, colCS3 = st.columns(3)
        chimney_small = colCS1.number_input("Small Chimney Removal (ea)", min_value=0, value=0, step=1)
        chimney_large = colCS2.number_input("Large Chimney Removal (ea)", min_value=0, value=0, step=1)
        skylights = colCS3.number_input("Skylight R&R (ea)", min_value=0, value=0, step=1)

    # Controls
    colCtrl1, colCtrl2, colCtrl3 = st.columns(3)
    waste_pct = colCtrl1.slider("Waste % (install items)", min_value=0, max_value=25, value=10, step=1)
    roofing_system = colCtrl2.selectbox("Roof System", ["architectural", "standing_seam", "exposed_fastener", "fortified"], index=0)
    ridge_vent_mode = colCtrl3.selectbox("Ridge Vent Mode", ["add", "cut"], index=0)

    st.markdown("**Insurance**")
    colIns1, colIns2, colIns3, colIns4 = st.columns(4)
    insurance_mode = colIns1.checkbox("Insurance job (apply O&P + deductible)", value=False)
    op_overhead_pct = colIns2.number_input("Overhead %", min_value=0.0, value=10.0, step=0.5)
    op_profit_pct = colIns3.number_input("Profit %", min_value=0.0, value=10.0, step=0.5)
    deductible = colIns4.number_input("Deductible ($)", min_value=0.0, value=0.0, step=100.0)

    if st.button("Calculate Estimate"):
        inputs = {
            "total_squares": total_squares,
            "total_area_sqft": total_area_sqft,
            "layers": layers,
            "ridge_lf": ridge_lf,
            "hip_lf": hip_lf,
            "valley_lf": valley_lf,
            "eave_lf": eave_lf,
            "rake_lf": rake_lf,
            "flashing_lf": flashing_lf,
            "ridge_vent_lf": ridge_vent_lf,
            # counts
            "pipe_boots": pipe_boots,
            "roof_vents": roof_vents,
            "exhaust_caps": exhaust_caps,
            "attic_fans": attic_fans,
            # chimney & skylights
            "chimney_small": chimney_small,
            "chimney_large": chimney_large,
            "skylights": skylights,
        }

        line_items, summary = compute_estimate(
            inputs,
            pricing,
            waste_pct=float(waste_pct),
            roofing_system=roofing_system,
            use_tiered_arch=use_tiered_arch,
            arch_small_rate=arch_small_rate,
            arch_large_rate=arch_large_rate,
            tier_break_sq=tier_break_sq,
            ridge_vent_mode=ridge_vent_mode,
            insurance_mode=insurance_mode,
            op_overhead_pct=float(op_overhead_pct),
            op_profit_pct=float(op_profit_pct),
            deductible=float(deductible),
        )

        st.subheader("Estimate")

        # Line items table
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
            st.info("No billable items detected. Add pricing and quantities to compute an estimate.")

        # Summary
        st.markdown("### Summary")
        st.write(
            f"Subtotal: **${summary['subtotal']:,.2f}**\n\n" +
            (f"O&P Overhead ({op_overhead_pct:.1f}%): **${summary['op_overhead']:,.2f}**\n\n" if insurance_mode else "") +
            (f"O&P Profit ({op_profit_pct:.1f}%): **${summary['op_profit']:,.2f}**\n\n" if insurance_mode else "") +
            f"Total before deductible: **${summary['total_before_deductible']:,.2f}**\n\n" +
            (f"Deductible: **-${summary['deductible']:,.2f}**\n\n" if insurance_mode else "") +
            f"**Total Due: ${summary['total_due']:,.2f}**"
        )

        st.caption("Notes: Tear-off calculated on raw squares × layers. Install items use waste-adjusted squares. Adjust O&P and deductible only if this is an insurance job.")
