import streamlit as st
import os
import tempfile
import langchain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# CHANGE IS HERE: Updated import for text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Page Setup
st.title("🧠 GenAI RAG Project")
st.caption("Upload a PDF and ask questions from it! (Powered by Gemma2:2b)")

# 2. Sidebar for Upload
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

# 3. Main Logic
if uploaded_file is not None:
    # --- STEP A: Load the PDF ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.success("File Uploaded! Processing...")
    
    # --- STEP B: Split Text (Chunking) ---
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    
    # Text ko chhote tukdon mein todna
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # --- STEP C: Create Embeddings (The Brain) ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # --- STEP D: Store in Vector DB (Chroma) ---
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    st.success("Analysis Complete! You can now chat below.")

    # --- STEP E: Setup Chat Interface ---
    
    # Model Setup (Gemma)
    llm = Ollama(model="gemma2:2b")
    
    # RAG Prompt (AI ko instruction)
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    # Chain banana (Connecting everything)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Chat UI Logic
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask something about your PDF..."):
        # User ka question dikhao
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # AI se answer maango
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
                try:
                    response = rag_chain.invoke({"input": user_input})
                    answer = response['answer']
                    st.markdown(answer)
                    # Answer save karo
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

    # Cleanup (Optional)
    try:
        os.remove(temp_file_path)
    except:
        pass

else:
    st.info("👈 Please upload a PDF from the sidebar to start!")