# FastAPI application for RAG-based QA (Bangla & English)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# Import RAG components
from document_loader import load_and_split
from vector_store import create_or_load_faiss
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
import numpy as np

# Defining request schema
class QARequest(BaseModel):
    question: str
    include_context: bool = False  # whether to return matched context

app = FastAPI(title="Bangla-English RAG QA API")

# --- Initialization (runs once on startup) ---
# Path to Bangla HSC book PDF (ensure embedded Unicode Bangla text)
PDF_PATH = "HSC26-Bangla1st-Paper.pdf"
FAISS_INDEX_DIR = "faiss_index"

# Load and chunk documents
documents = load_and_split(PDF_PATH, chunk_size=1000, chunk_overlap=100)
# Create or load FAISS vector store with multilingual embeddings
faiss_store = create_or_load_faiss(documents, index_dir=FAISS_INDEX_DIR)
# Set up Retriever for semantic search (top K=3 chunks)
retriever = faiss_store.as_retriever(search_kwargs={"k": 3})
# Initialize Ollama LLM (Mistral model)
llm = Ollama(model="mistral")  # Ensure Ollama and the Mistral model are installed locally
# Create RetrievalQA chain (combines retriever + LLM)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True)

# SentenceTransformer for similarity calculations
embed_model = SentenceTransformer('intfloat/multilingual-e5-base')



"""
    /ask endpoint:
    - Accepts question in Bangla or English.
    - Retrieves top-K relevant chunks via FAISS semantic search.
    - Generates answer with Mistral LLM (via Ollama) using the retrieved.
    - Returns the answer and optionally the matched context with similarity scores.
    """
    
@app.post("/ask")
def ask_question(req: QARequest):
    
    query = req.question
    if not query:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Retrieve relevant documents (chunks)
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return {"answer": "", "contexts": [], "similarities": [], "groundedness": 0.0}

    # Combine chunks into context string for the LLM
    context_texts = [doc.page_content for doc in docs]
    context = "\n\n".join(context_texts)

    # Generate answer using the RAG chain (includes retrieved context internally).
    # The chain returns a dict with 'result' (answer) and 'source_documents' if requested.
    result = qa_chain({"query": query})
    answer = result["result"]
    source_docs = result.get("source_documents", [])

    # Compute cosine similarities between query and each retrieved.
    # Use normalized embeddings: dot product equals cosine similarity.
    query_vec = embed_model.encode([query], normalize_embeddings=True)[0]
    doc_vecs = embed_model.encode(context_texts, normalize_embeddings=True)
    similarities = [float(np.dot(query_vec, doc_vec)) for doc_vec in doc_vecs]

    # Basic groundedness: average cosine similarity between answer and context chunks
    answer_vec = embed_model.encode([answer], normalize_embeddings=True)[0]
    # If no context vectors, set groundedness to 0. Otherwise, average cosines.
    if len(doc_vecs) > 0:
        cosines = [float(np.dot(answer_vec, vec)) for vec in doc_vecs]
        groundedness = sum(cosines) / len(cosines)
    else:
        groundedness = 0.0

    response = {"answer": answer, "similarities": similarities, "groundedness": groundedness}
    if req.include_context:
        # Include matched context chunks
        response["contexts"] = context_texts

    return response


# Run app with: uvicorn app:app --reload --port 3800