from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader #requires bsoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from settings import DEFAULT_URLS, DEFAULT_PDFS # data sources

# -------------------------------------------------
# Helper functions for handling document ingestion, splitting, and vector store creation
# -------------------------------------------------


### LOAD ###
def load_websites_data(urls):
    """
    Load documents from a list of websites.

    Args:
        urls (list): List of website URLs.

    Returns:
        list: Combined documents from all websites.
    """
    web_documents = []
    for url in urls:
        loader = WebBaseLoader(url)
        document = loader.load()
        web_documents.extend(document)

    return web_documents


def load_pdfs_data(pdfs):
    """
    Load documents from a list of PDF files.

    Args:
        pdfs (list): List of file paths to PDF documents.

    Returns:
        list: Combined documents from all PDFs.
    """
    pdf_documents = []
    for pdf in pdfs:
        loader = PyPDFLoader(pdf)
        document = loader.load()
        pdf_documents.extend(document) 

    return pdf_documents


def load_all_data_sources():
    """
    Load all documents from both web and PDF sources.

    Returns:
        list: Combined list of documents from all data sources.
    """
    all_web_documents = load_websites_data(DEFAULT_URLS)
    all_pdf_documents = load_pdfs_data(DEFAULT_PDFS)
    all_documents = all_pdf_documents + all_web_documents
    
    return all_documents


### SPLIT ###
def split_documents(documents):
    """Dividir todos los documentos combinados en fragmentos."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    all_splitted_documents = text_splitter.split_documents(documents)
    
    return all_splitted_documents
  

### STORE ###
def get_vector_store(documents_chunks):
    """
    Create a FAISS vector store from document chunks.

    Args:
        document_chunks (list): List of document chunks to embed and store.

    Returns:
        FAISS: A FAISS vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_documents(documents_chunks, embeddings)
    
    # Save vector store locally (optional)
    vector_store.save_local("faiss_index")
    
    return vector_store
