import os  
import fitz  # PyMuPDF  
import openai  
import streamlit as st  
from pinecone import Pinecone, ServerlessSpec  
  
# Initialize Pinecone  
pc = Pinecone(  
    api_key="YOUR_PINECONE_API_KEY"  # Replace with your Pinecone API key  
)  
  
# Check if the index exists, and create it if it doesn't  
index_name = "newspaper-embeddings"  # Replace with your index name  
if index_name not in pc.list_indexes().names():  
    pc.create_index(  
        name=index_name,  
        dimension=1536,  # Dimension of the embedding (e.g., for text-embedding-ada-002)  
        metric="cosine",  # Similarity metric (e.g., cosine, euclidean)  
        spec=ServerlessSpec(  
            cloud="aws",  # Replace with your cloud provider (e.g., aws, gcp)  
            region="us-west-2"  # Replace with your Pinecone region  
        )  
    )  
  
# Connect to the index  
index = pc.index(index_name)  
  
# Function to generate embeddings using OpenAI API  
def get_embedding(text, model="text-embedding-ada-002"):  
    response = openai.Embedding.create(  
        input=text,  
        model=model  
    )  
    return response['data'][0]['embedding']  
  
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
st.title("PDF Embedding Processor")  
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)  
  
if uploaded_files:  
    for uploaded_file in uploaded_files:  
        st.write(f"Processing {uploaded_file.name}...")  
        process_pdf(uploaded_file)  
    st.write("All PDFs processed and embeddings stored in Pinecone.")  
