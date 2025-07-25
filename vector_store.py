# Vector Store Setup with FAISS and multilingual embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

"""
    Create or load a FAISS vector store with persistence.
    Embedding model: intfloat/multilingual-e5-base (multilingual, 94 languages support).
"""

def create_or_load_faiss(documents, index_dir: str = "faiss_index"):
    
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    # Load existing index if available
    if os.path.exists(index_dir):
        faiss_store = FAISS.load_local(index_dir, embedding_model)
    else:
        # Create new FAISS index from documents and save it
        faiss_store = FAISS.from_documents(documents, embedding_model)
        faiss_store.save_local(index_dir)
    return faiss_store
