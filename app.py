# rep_gpt_app.py

import streamlit as st
import openai

# Set your API key (optionally, use Streamlit secrets instead)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit app config
st.set_page_config(page_title="Rep GPT Assistant", page_icon="🛠️")
st.title("🛠️ Rep GPT Assistant")
st.caption("Your AI-powered field sales and roofing helper.")

# User input
user_input = st.text_area("Ask a question:", placeholder="Example: How do I explain the difference between shingles?", height=100)

# Response area
if st.button("Send") and user_input.strip():
    with st.spinner("Getting response..."):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """
You are RepBot, a knowledgeable and helpful AI assistant for Pro Roofing field reps.

Your job is to:
- Answer product and service questions clearly.
- Help reps explain roofing options, materials, and warranties to customers.
- Assist with estimating, scheduling, and sales guidance.
- Write short, professional messages reps can send to customers.
- Keep answers under 100 words unless more is asked for.
- Avoid technical jargon—keep it simple and friendly.

Always speak in a respectful, confident, and helpful tone.
"""}, 
                    {"role": "user", "content": user_input}
                ]
            )
            reply = response.choices[0].message.content
            st.markdown("### 💬 Response")
            st.write(reply)
        except Exception as e:
            st.error(f"Error: {str(e)}")
