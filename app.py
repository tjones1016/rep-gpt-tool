import streamlit as st
import openai

# Set your API key here or use secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Page config
st.set_page_config(page_title="Rep GPT Assistant", page_icon="🤖")
st.title("🛠️ Rep GPT Assistant")
st.caption("Ask anything related to your workflow.")

# Input box
user_input = st.text_area("Enter your question:", height=100)

# Send request to OpenAI
if st.button("Send") and user_input.strip():
    with st.spinner("Thinking..."):
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for field sales and roofing reps. Keep answers short, clear, and action-focused."},
                {"role": "user", "content": user_input}
            ]
        )
        reply = response.choices[0].message.content
        st.markdown("### 💬 GPT Response")
        st.write(reply)
