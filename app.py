import os
import streamlit as st
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# --- Hard-coded rules ---
HARD_CODED_KNOWLEDGE = """
SLAP Formula:
- Say hi and introduce yourself
- Let them know why you’re here
- Ask an open-ended question
- Present your pitch

ARO Formula:
- Acknowledge
- Reassure
- Overcome

Pricing, Payment, Accountability:
- Pricing: Adjusters use Xactimate for all estimates
- Payment: 1st payment, deductible, 2nd payment
- Accountability: Homeowner saves no money, the carrier controls the process, use the body shop analogy to communicate simplicity

Uncertainty Rule:
- If you’re unsure whether there’s enough roof damage, end with: "File the Claim — let the adjuster make the call."
"""

PRICING_DISCLAIMER = (
    "Always confirm with Xactimate if an adjuster asks about pricing. "
    "These prices are for rep reference only."
)

# --- Load documents into FAISS ---
@st.cache_resource
def load_vectorstore():
    loaders = [
        Docx2txtLoader("data/sales_guide.docx"),
        Docx2txtLoader("data/sales_guide3.docx"),
        PyPDFLoader("data/d2d_script.pdf"),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(split_docs, embeddings)

# --- Load pricing CSV ---
@st.cache_resource
def load_pricing():
    try:
        df = pd.read_csv("data/pricing.csv")
        return df
    except Exception as e:
        st.error(f"Error loading pricing.csv: {e}")
        return pd.DataFrame()

def lookup_price(query, df):
    query_lower = query.lower()
    for _, row in df.iterrows():
        item_name = str(row["Item"]).lower()
        if item_name in query_lower:
            return f"{row['Item']}: {row['Price']} per {row['Unit']}. Notes: {row.get('Notes','')}\n\n{PRICING_DISCLAIMER}"
    return None

# --- Setup conversational chain ---
@st.cache_resource
def setup_conversational_chain(vstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vstore.as_retriever(search_kwargs={"k": 3})
    return ConversationalRetrievalChain.from_llm(llm, retriever)

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Pro Roofing AI Assistant", page_icon=":construction:")
    st.title("🦺 Pro Roofing AI Assistant")

    vstore = load_vectorstore()
    qa_chain = setup_conversational_chain(vstore)
    pricing_df = load_pricing()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.chat_input("Ask me something...")
    if user_query:
        # 1. Check pricing first
        pricing_answer = lookup_price(user_query, pricing_df)

        if pricing_answer:
            bot_response = pricing_answer
        else:
            # 2. Query vectorstore
            result = qa_chain.invoke({"question": user_query, "chat_history": st.session_state.chat_history})
            bot_response = result["answer"]

            # 3. Add hard-coded rules
            bot_response += "\n\n" + HARD_CODED_KNOWLEDGE

        # Update chat history
        st.session_state.chat_history.append(("User", user_query))
        st.session_state.chat_history.append(("AI", bot_response))

    # Display conversation
    for speaker, text in st.session_state.chat_history:
        if speaker == "User":
            st.chat_message("user").markdown(text)
        else:
            st.chat_message("assistant").markdown(text)

if __name__ == "__main__":
    main()
