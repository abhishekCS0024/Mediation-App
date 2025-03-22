import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings  # Changed import to community version
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv  # Uncommented for environment loading

# Load environment variables properly
load_dotenv()  # Uncommented to ensure environment variables are loaded

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("Error: HF_TOKEN not found in environment variables")
    exit(1)

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            model_kwargs={
                "token": HF_TOKEN,
                "max_length": 512  # Changed to integer instead of string
            }
        )
        return llm
    except Exception as e:
        print(f"Error initializing HuggingFace model: {str(e)}")
        exit(1)

# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context
Context: {context}
Question: {question}
Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
if not os.path.exists(DB_FAISS_PATH):
    print(f"Error: Vector store not found at {DB_FAISS_PATH}")
    print("Please run the ingest.py script first to process your PDF documents.")
    exit(1)

try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Error loading vector store: {str(e)}")
    exit(1)

# Create QA chain
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(HUGGINGFACE_REPO_ID),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
except Exception as e:
    print(f"Error creating QA chain: {str(e)}")
    exit(1)

# Now invoke with a single query
try:
    user_query = input("Write Query Here: ")
    response = qa_chain.invoke({'query': user_query})
    print("\nRESULT: ", response["result"])
    print("\nSOURCE DOCUMENTS: ")
    for i, doc in enumerate(response["source_documents"]):
        print(f"\nDocument {i+1}:")
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
except Exception as e:
    print(f"Error processing query: {str(e)}")