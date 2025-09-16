import os
import re
import io
from typing import Dict, Any, Tuple

import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# System prompt with hard-coded rules
# -----------------------------
system_prompt = """
You are the Pro-Roofing AI Sales Assistant. You are trained on the company’s sales guide,
compensation plan, SOPs, and related documents. Your role is to help sales representatives 
by answering questions, reinforcing training, and reviewing important sales formulas and steps.

Always remember and be ready to explain the following:

1. SLAP Formula:
   - S: Say hi and introduce yourself
   - L: Let them know why you’re here
   - A: Ask an open-ended question
   - P: Present your pitch

2. ARO Formula:
   - A: Acknowledge
   - R: Reassure
   - O: Overcome

3. After the homeowner files the claim, the rep must explain:
   - Pricing:
     • Adjusters use Xactimate, not the contractor, to set pricing.
   - Payment:
     • 1st payment (ACV/initial check)
     • Deductible (homeowner’s responsibility)
     • 2nd payment (recoverable depreciation/final check)
   - Accountability:
     • The carrier is in control of the process, not the contractor.
     • There is no money saved by choosing a cheaper contractor.
     • Use the "body shop" analogy: just like car repairs after an accident,
       the insurance company pays the set amount and the contractor does the work.

4. If a rep is unsure whether there is enough damage on a roof:
   - Encourage the homeowner to "File the Claim"
   - Remind them: the adjuster makes the final call, not the rep.

Always provide clear, confident, and supportive answers, and tie them back to these
frameworks whenever relevant.
"""

# -----------------------------
# Document loading
# -----------------------------
def load_all_docs(data_dir="data"):
    docs = []
    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        if file.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif file.lower().endswith(".docx"):
            docs.extend(Docx2txtLoader(path).load())
        elif file.lower().endswith(".txt"):
            docs.extend(TextLoader(path).load())
    return docs

# -----------------------------
# Build or load vectorstore
# -----------------------------
def build_vectorstore():
    persist_directory = "vectorstore"
    if os.path.exists(persist_directory):
        return FAISS.load_local(
            persist_directory,
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        )

    docs = load_all_docs("data")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    vectorstore.save_local(persist_directory)
    return vectorstore

# -----------------------------
# RAG pipeline
# -----------------------------
def get_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever()

    template = """You are the Pro-Roofing AI Sales Assistant.
Follow the system rules and frameworks provided.

System Rules:
{system_prompt}

Context from training docs:
{context}

User Question:
{question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough(), "system_prompt": lambda _: system_prompt}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Pro-Roofing AI Assistant", page_icon="🏠")
    st.header("Pro-Roofing AI Sales Assistant")

    vectorstore = build_vectorstore()
    rag_chain = get_rag_chain(vectorstore)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask the Sales Assistant a question")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        response = rag_chain.invoke(user_input)
        st.session_state.chat_history.append(("assistant", response))

    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(text)

if __name__ == "__main__":
    main()
