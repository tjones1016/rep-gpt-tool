import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Page config (title + favicon/logo)
st.set_page_config(
    page_title="Rep GPT Assistant",
    page_icon="apple-touch-icon.png",  # 👈 Your company logo file
)

st.title("🛠️ Rep GPT Assistant")
st.caption("Ask me anything based on our training material!")

# Inject meta tag so iOS uses your icon when saving to Home Screen
st.markdown(
    """
    <link rel="apple-touch-icon" href="apple-touch-icon.png" />
    """,
    unsafe_allow_html=True,
)

# Load vectorstore (replace with your real FAISS index path)
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# Prompt template (your existing QA logic stays unchanged)
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant for our sales reps.
Use the following context to answer the question as accurately as possible.

Context:
{context}

Question:
{question}

Answer in detail:
"""
)

# Setup chain (fixed with ChatOpenAI wrapper)
def setup_conversational_chain():
    llm = ChatOpenAI(
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        model="gpt-4o-mini",  # or "gpt-4o" if you prefer
        temperature=0
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )

qa_chain = setup_conversational_chain()

# UI chat interface
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me something..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result = qa_chain.invoke({"question": prompt})
        response = result["answer"]
        st.markdown(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})
