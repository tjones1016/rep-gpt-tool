import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --------------------------
# SETTINGS
# --------------------------
DATA_FOLDER = "data"
EMBEDDINGS_FILE = "vectorstore/faiss_index"

# Initialize model + embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------
# HARDCODED RULES
# --------------------------
HARDCODED_RULES = {
    "slap": "SLAP stands for Stop, Look, Ask, Pause. This is the only definition Pro-Roofing uses.",
    "aro": "ARO means Ask for the Referral Opportunity. Always ask for referrals at the right time in the conversation.",
    "pricing": "Pricing/Payment/Accountability: Reps must follow the company’s pricing guide, review payment terms with the homeowner, and maintain accountability in AccuLynx.",
    "file the claim": "File-the-Claim: If the homeowner is ready, walk them through calling their insurance carrier to file the claim right away."
}

def check_hardcoded_rules(query: str):
    q_lower = query.lower()
    for key, response in HARDCODED_RULES.items():
        if key in q_lower:
            return response
    return None

# --------------------------
# LOAD VECTORSTORE
# --------------------------
@st.cache_resource
def load_vectorstore():
    if os.path.exists(EMBEDDINGS_FILE):
        return FAISS.load_local(EMBEDDINGS_FILE, embeddings, allow_dangerous_deserialization=True)

    docs = []
    for file in os.listdir(DATA_FOLDER):
        filepath = os.path.join(DATA_FOLDER, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            docs.extend(loader.load())
        elif file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filepath)
            docs.extend(loader.load())
        elif file.endswith(".csv"):
            loader = CSVLoader(filepath)
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(EMBEDDINGS_FILE)
    return vectorstore

vectorstore = load_vectorstore()

# Conversational Retrieval
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# --------------------------
# STREAMLIT APP
# --------------------------
st.set_page_config(page_title="Pro-Roofing AI Assistant", page_icon="🦺")

st.image("apple-touch-icon.png", width=120)
st.title("Pro-Roofing AI Assistant")

user_query = st.chat_input("Ask me anything about sales, pricing, or procedures...")

if user_query:
    # First check hardcoded rules
    hardcoded_response = check_hardcoded_rules(user_query)
    if hardcoded_response:
        st.chat_message("assistant").write(hardcoded_response)
    else:
        # Otherwise query vectorstore + LLM
        response = qa_chain.run(user_query)
        st.chat_message("assistant").write(response)
