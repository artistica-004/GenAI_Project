import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="GenAI Master Project", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🤖 GenAI Project")
st.sidebar.caption("Created by a Future GenAI Engineer")

# Ab yahan 4 Options hain
app_mode = st.sidebar.selectbox(
    "Choose a Feature:", 
    ["🏠 Home", "💬 1. Simple Chatbot", "🔍 2. Document Analysis", "🧠 3. RAG System (Chat with PDF)"]
)

# --- 1. HOME PAGE ---
if app_mode == "🏠 Home":
    st.title("🚀 My Generative AI Project")
    st.markdown("""
    ### Welcome to my Portfolio Project!
    This application demonstrates the complete lifecycle of a GenAI project.
    
    #### Modules Explained:
    - **💬 1. Simple Chatbot:** Basic AI Agent connecting to Gemma2:2b.
    - **🔍 2. Document Analysis:** Raw data extraction pipeline to read and understand PDFs.
    - **🧠 3. RAG System:** Advanced Retrieval-Augmented Generation to chat with documents.
    
    #### Tech Stack:
    - Python, Streamlit, LangChain, Ollama, ChromaDB.
    
    👈 **Select a module from the Sidebar to start!**
    """)

# --- 2. SIMPLE CHATBOT (Phase 1) ---
elif app_mode == "💬 1. Simple Chatbot":
    st.header("💬 Simple Chatbot")
    st.caption("This model runs locally without internet.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "Hi! I am Gemma. Ask me anything."}]

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Type your message..."):
        st.chat_message("user").write(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = Ollama(model="gemma2:2b")
                    response = llm.invoke(prompt)
                    st.write(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}. Make sure Ollama is running.")

# --- 3. DOCUMENT ANALYSIS (Phase 2 - Wapas aa gaya!) ---
elif app_mode == "🔍 2. Document Analysis":
    st.header("🔍 Document Analysis Engine")
    st.caption("This module extracts and analyzes raw text from PDFs (No Chat, Just Analysis).")

    uploaded_file = st.file_uploader("Upload a PDF for Analysis", type="pdf")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        if st.button("Analyze Document"):
            with st.spinner("Extracting Data..."):
                try:
                    loader = PyPDFLoader(temp_file_path)
                    docs = loader.load()
                    
                    st.success(f"Analysis Complete! Found {len(docs)} pages.")
                    
                    # Show Preview of Text
                    st.subheader("📄 Extracted Content Preview:")
                    for i, doc in enumerate(docs[:3]):  # Sirf pehle 3 page dikhayega
                        with st.expander(f"Page {i+1} Content"):
                            st.write(doc.page_content)
                            
                    st.info("Note: This raw text is what sends to the LLM in the next stage (RAG).")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Cleanup
        try:
            os.remove(temp_file_path)
        except:
            pass

# --- 4. RAG SYSTEM (Phase 3) ---
elif app_mode == "🧠 3. RAG System (Chat with PDF)":
    st.header("🧠 RAG System (Chat with PDF)")
    st.caption("Upload a PDF and ask questions. The AI will answer ONLY from the document.")

    uploaded_file = st.file_uploader("Upload PDF for RAG", type="pdf")

    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Process PDF only once
        if "processed_file_rag" not in st.session_state or st.session_state.processed_file_rag != uploaded_file.name:
            with st.spinner("Creating Vector Embeddings..."):
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                
                st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                st.session_state.processed_file_rag = uploaded_file.name
                st.success("Database Ready! Ask your questions below.")

        # Chat Interface
        for msg in st.session_state.rag_history:
            st.chat_message(msg["role"]).write(msg["content"])

        if user_question := st.chat_input("Ask about the PDF..."):
            st.chat_message("user").write(user_question)
            st.session_state.rag_history.append({"role": "user", "content": user_question})

            with st.chat_message("assistant"):
                with st.spinner("Searching document..."):
                    try:
                        retriever = st.session_state.vectorstore.as_retriever()
                        llm = Ollama(model="gemma2:2b")
                        
                        system_prompt = (
                            "You are an assistant. Answer the question based ONLY on the following context. "
                            "If the answer is not in the context, say 'I don't know based on this document'."
                            "\n\nContext: {context}"
                        )
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            ("human", "{input}"),
                        ])
                        question_answer_chain = create_stuff_documents_chain(llm, prompt)
                        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                        response = rag_chain.invoke({"input": user_question})
                        answer = response['answer']
                        
                        st.write(answer)
                        st.session_state.rag_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error: {e}")

        try:
            os.remove(temp_file_path)
        except:
            pass