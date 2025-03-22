from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define paths
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_pdf_files(data_path):
    """Load PDF files from a directory"""
    print(f"Loading PDF files from {data_path}")
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} PDF pages")
    return documents

def create_chunks(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into chunks"""
    print(f"Creating chunks with size {chunk_size} and overlap {chunk_overlap}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(documents)
    print(f"Created {len(text_chunks)} text chunks")
    return text_chunks

def create_vector_store(text_chunks):
    """Create vector embeddings and store in FAISS"""
    print("Creating embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    
    # Create and save FAISS index
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store saved to {DB_FAISS_PATH}")
    return db

def main():
    print("Starting document ingestion process...")
    
    # Make sure data directory exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created data directory at {DATA_PATH}")
        print("Please add PDF files to this directory and run this script again.")
        return
    
    # Check if there are PDFs in the data directory
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {DATA_PATH}. Please add some PDF files and try again.")
        return
    
    print(f"Found {len(pdf_files)} PDF files: {', '.join(pdf_files)}")
    
    # Process steps
    documents = load_pdf_files(DATA_PATH)
    text_chunks = create_chunks(documents)
    create_vector_store(text_chunks)
    print("Document ingestion completed successfully!")

if __name__ == "__main__":
    main()