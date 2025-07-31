import streamlit as st
import openai

# Initialize OpenAI client using new SDK structure
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Streamlit page config
st.set_page_config(page_title="Rep GPT Assistant", page_icon="🛠️")
st.title("🛠️ Rep GPT Assistant")
st.caption("Your AI-powered field sales and roofing helper.")

# User input area
user_input = st.text_area(
    "Ask a question:",
    placeholder="Example: How do I explain the difference between shingles?",
    height=100
)

# Generate GPT response
if st.button("Send") and user_input.strip():
    with st.spinner("Thinking..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """
You are RepBot, a knowledgeable and helpful AI assistant for Pro Roofing field reps.

Your job is to:
- Answer product and service questions clearly.
- Help reps explain roofing options, materials, and warranties to customers.
- Assist with estimating, scheduling, and sales guidance.
- Write short, professional messages reps can send to customers.
- Keep answers under 100 words unless asked for more.
- Avoid technical jargon—keep it simple and friendly.

Always speak in a respectful, confident, and helpful tone.
                        """.strip()
                    },
                    {"role": "user", "content": user_input}
                ]
            )
            reply = response.choices[0].message.content.strip()
            st.markdown("### 💬 GPT Response")
            st.write(reply)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
