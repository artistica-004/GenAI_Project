import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

st.title("📄 PDF Reader (Step 2)")
st.caption("Upload a PDF and I will extract the text for you.")

# 1. File Upload Button
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # 2. Temp file banana padta hai kyunki PyPDFLoader direct memory se nahi padhta
    # Yeh thoda technical hai, bas maan lo ki hum file ko temporary save kar rahe hain
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.success("File uploaded successfully!")
    
    # 3. PDF se Text nikalna (Extract)
    st.write("Extracting text...")
    
    try:
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        
        # 4. Text ko screen pe dikhao
        st.subheader("What I found in the PDF:")
        for i, doc in enumerate(docs):
            st.write(f"**Page {i+1}:**")
            st.write(doc.page_content)
            st.markdown("---") # Line create karne ke liye
            
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        
    finally:
        # 5. Safai Abhiyan: Kaam hone ke baad temp file delete kar do
        os.remove(temp_file_path)