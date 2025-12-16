# 🧠 End-to-End GenAI Project: Chatbot & RAG Pipeline

This project is a **Generative AI application** built to demonstrate the complete lifecycle of an LLM-based application. It allows users to chat with an AI and perform **Retrieval-Augmented Generation (RAG)** on their private PDF documents locally.

## 🚀 Features
- **Simple Chatbot:** A conversational agent powered by **Gemma2:2b** (running locally via Ollama).
- **Document Analysis:** Extracts raw text from uploaded PDFs for data inspection.
- **RAG System:** Enables "Chat with PDF" functionality using Vector Embeddings.
- **Privacy First:** Runs 100% locally on the user's machine. No data is sent to the cloud.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Frontend:** Streamlit
- **LLM Orchestration:** LangChain
- **Vector Database:** ChromaDB
- **Model Provider:** Ollama (Gemma2:2b)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)

## ⚙️ How to Run Locally

1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/your-username/GenAI_Project.git](https://github.com/your-username/GenAI_Project.git)
