# Dependencies: pip install streamlit langchain langchain-openai beautifulsoup4 python-dotenv chromadb
import os
import re
from pathlib import Path
from typing import Callable, Union, List, Dict, Any

# Environment setup
from dotenv import load_dotenv

# Langserve for FastAPI integration
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

# Langchain modules
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory, ChatMessageHistory

# Custom utilities for RAG (Retrieval-Augmented Generation)
from rag_utils import load_all_data_sources, split_documents, get_vector_store
#from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader #requires bsoup
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.vectorstores import FAISS
from rag_utils import load_all_data_sources, split_documents, get_vector_store

### Load environment variables ###
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
MODEL_NAME = "llama3-8b-8192" #'mixtral-8x7b-32768'



# -------------------------------------------------
# Helper functions for managing chat histories
# -------------------------------------------------

### In-memory chat history ###
store_chat_history = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve in-memory chat history."""
    if session_id not in store_chat_history:
        store_chat_history[session_id] = ChatMessageHistory()
    return store_chat_history[session_id]



# -------------------------------------------------
# RAG Configuration
# -------------------------------------------------

### Load and process documents ###
all_documents = load_all_data_sources()
print(f"Total documentos combinados: {len(all_documents)}")

document_chunks = split_documents(all_documents)
print(f"Total fragmentos generados: {len(document_chunks)}")

vector_store = get_vector_store(document_chunks)
print("Vector store creado y guardado localmente.")


### Configure retriever (Given a user input, relevant chunks are retrieved from storage using a Retriever) ###

retriever = vector_store.as_retriever() # Give us the documents that are relevant to our conversation

contextualize_q_system_prompt2 = """
Given a chat history and the latest user question which might reference context in the chat history, 
formulate a standalone question which can be understood without the chat history. 
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""

contextualize_q_system_prompt = """
You are a search query expert. Your task is to:
    1. Analyze if the question requires information from our knowledge base
    2. Create a relevant search query only if needed
    3. Focus on key concepts rather than exact phrases
"""

retriever_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME
)

retriever_chain = create_history_aware_retriever(
    llm, retriever, retriever_prompt
)
# Create a chain that takes conversation history and returns documents.
# - If there is no chat_history, then the input is just passed directly to the retriever. 
# - If there is chat_history, then the prompt and LLM will be used to generate a search query. 
#   That search query is then passed to the retriever. 
# https://api.python.langchain.com/en/latest/chains/langchain.chains.history_aware_retriever.create_history_aware_retriever.html
print("Cadena de recuperación configurada.")


### Configure RAG chain for generation ###

# Prompt for question-answering:
qa_system_prompt = """You are a knowledgeable AI assistant. For each question:
    1. First evaluate if the retrieved context is actually relevant
    2. For relevant context, use it as your primary source
    3. For general knowledge questions, rely on your base knowledge
    4. You can combine both sources when appropriate
    5. Always maintain a natural, conversational tone
    6. If you don't have enough information to answer accurately, acknowledge this and suggest what additional information would be helpful
    
Context provided:
{context}

Remember: Not every question needs to be answered using the context."""

response_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

stuff_documents_chain = create_stuff_documents_chain(
    llm, response_prompt
) # Create a chain for passing a list of Documents to a model

conversational_rag_chain = create_retrieval_chain(
    retriever_chain, stuff_documents_chain
)
print("Cadena de generación configurada.")



# -------------------------------------------------
# LangServe FastAPI Server
# -------------------------------------------------

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# CORS settings for API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add chat routes with session-based RAG chain
rag_chain = RunnableWithMessageHistory(
    conversational_rag_chain,
    #create_session_factory("chat_rag_histories"),
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

add_routes(
    app, rag_chain, path="/chat",
)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
