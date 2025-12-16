# Import necessary libraries
import streamlit as st
from langchain_community.llms import Ollama

# 1. Title aur Page Setup
st.title("🤖 My GenAI Chatbot (Gemma2:2b)")
st.caption("A local chatbot built by [Your Name]")

# 2. Sidebar - User se Model select karwana (Optional but looks cool)
with st.sidebar:
    st.header("Settings")
    st.write("This bot uses a locally running LLM.")
    model_id = "gemma2:2b"

# 3. Chat History Setup
# Hum check kar rahe hain ki kya pehle se koi chat saved hai? Agar nahi, to empty list banao.
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

# 4. Purani Chat ko Screen pe dikhao (Loop)
for msg in st.session_state.messages:
    # 'chat_message' UI banata hai (user ka icon ya robot ka icon)
    st.chat_message(msg["role"]).write(msg["content"])

# 5. User Input Handling
# Jab user box me kuch likhega:
if prompt := st.chat_input("Type your question here..."):
    
    # User ka message screen pe dikhao
    st.chat_message("user").write(prompt)
    
    # User ka message history me save karo
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 6. Generate Response (AI ka Jawab)
    # Spinner dikhao jab tak AI soch raha hai
    with st.spinner("Thinking..."):
        try:
            # Ollama (Gemma model) ko initialize karo
            llm = Ollama(model=model_id)
            
            # AI se answer maango
            response = llm.invoke(prompt)
            
            # AI ka answer screen pe dikhao
            st.chat_message("assistant").write(response)
            
            # AI ka answer history me save karo
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"Error: {e}. Make sure Ollama is running!")

'''Step 3: Explanation (Interview ke liye)Agar interviewer puche "Code kaise kaam karta hai?",
toh yeh points bolo:Streamlit (st): Maine use kiya frontend banane ke liye.
Yeh st.chat_message function use karta hai chat bubble dikhane ke liye.
Session State: Maine st.session_state use kiya taaki jab main naya message likhu, toh purani chat delete na ho jaye (It remembers history).
Ollama Integration: Maine langchain_community library use ki taaki main Python se mere local Ollama model ko connect kar saku.
Flow: User input $\rightarrow$ Session State update $\rightarrow$ Send to LLM $\rightarrow$ Get Response $\rightarrow$ Update UI.Step 4: Run the Project'''