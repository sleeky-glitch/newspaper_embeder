import os  
import fitz  # PyMuPDF  
import pinecone  
import json  
import streamlit as st  
from openai.embeddings_utils import get_embedding  

# Streamlit UI  
st.title("PDF Embedding Processor")  
st.write("Upload your PDFs to process and store embeddings in Pinecone.")  

# Read API keys from Streamlit secrets  
pinecone_api_key = st.secrets["api_keys"]["pinecone_api_key"]  
pinecone_environment = st.secrets["api_keys"]["pinecone_environment"]  
openai_api_key = st.secrets["api_keys"]["openai_api_key"]  

# Initialize Pinecone and OpenAI  
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)  
openai.api_key = openai_api_key  


# Function to process PDFs and generate embeddings  
def process_pdf(file):  
    doc = fitz.open(stream=file.read(), filetype="pdf")  
    for page_number, page in enumerate(doc, start=1):  
        text = page.get_text()  
        if text.strip():  # Only process pages with text  
            # Generate embedding  
            embedding = get_embedding(text, model="text-embedding-ada-002")  

            # Store embedding in Pinecone  
            metadata = {  
                "file_name": file.name,  
                "page_number": page_number,  
                "content": text,  
            }  
            index.upsert([(f"{file.name}_page_{page_number}", embedding, metadata)])  
    st.success(f"Finished processing {file.name}.")  

# Streamlit file uploader  
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)  

if uploaded_files:  
    for uploaded_file in uploaded_files:  
        st.write(f"Processing {uploaded_file.name}...")  
        process_pdf(uploaded_file)  
    st.write("All PDFs processed and embeddings stored in Pinecone.")  
