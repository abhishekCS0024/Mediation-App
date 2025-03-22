

import os
import streamlit as st
# Updated import for HuggingFaceEmbeddings from the correct package
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import time

# For Groq
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"

# Groq model options
GROQ_MODEL = "llama3-70b-8192"  # Default model

# Custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
chatbot start with a question : "Start"
for hi: write what you want to ask 

# understand the question is it about the context if not , don't use vector store database


Also try to capture the context from the question and provide the answer accordingly
If required Create a SOLUTION section about their problem 
Mention the acts & related case to it
Mention the list of steps to possibly resolve the conflict without going to higher authorities



Context: {context}
Question: {question}


"""

# understand the question is it about the context if not , don't use vector store database
# give answer to the question based on the context provided. If you dont know the answer, just say that you dont know, dont try to make up an answer. Dont provide anything out of the given context
# Start your answer directly without any small talk or introductions.
# Load and configure CSS
def configure_page():
    st.set_page_config(
        page_title="Mediation Q&A Bot",
        page_icon="📚",
        layout="centered"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .message {
        width: 80%;
        padding-left: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Get the vector store
@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        st.error(f"Failed to load vector store: {str(e)}")
        return None

# Create prompt template
def set_custom_prompt():
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE, 
        input_variables=["context", "question"]
    )
    return prompt

# Load the language model
def load_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
        return None
    
    try:
        return ChatGroq(
            groq_api_key=groq_api_key,
            model_name=GROQ_MODEL,
            temperature=0.5,
            max_tokens=1024
        )
    except Exception as e:
        st.error(f"Error initializing Groq model: {str(e)}")
        return None

# Create QA chain
def get_qa_chain():
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return None
        
    llm = load_llm()
    if llm is None:
        return None
    
    try:    
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 4}),
            return_source_documents=False,  # Don't return source documents
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

# Main app function
def main():
    configure_page()
    
    st.title("📚 Mediation Q&A Bot")
    
    # Check if vector store exists
    if not os.path.exists(DB_FAISS_PATH):
        st.warning("Vector store not found. Please run the ingest.py script first to process your PDF documents.")
        st.info("Run `python ingest.py` after adding PDF files to the 'data/' directory.")
        return
    
    # Add information about the app
    st.markdown("You can discuss about the conflicts & disagreements. To intiate it type 'Start' in the chatbox.")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    # Create QA chain
                    qa_chain = get_qa_chain()
                    if qa_chain is None:
                        message_placeholder.error("Failed to initialize the QA chain. Please check your API keys and configuration.")
                        return
                    
                    # Get response
                    start_time = time.time()
                    response = qa_chain.invoke({"query": prompt})
                    elapsed_time = time.time() - start_time

                    # Extract result
                    result = response["result"]
                    
                    # Display response
                    message_placeholder.markdown(result)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                except Exception as e:
                    message_placeholder.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()