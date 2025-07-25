# Document Loader and Chunker for Bangla HSC PDF
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


"""
    Load a PDF and split into text chunks (sentences/paragraphs).
    Uses LangChain's PyPDFLoader and CharacterTextSplitter.
    Returns a list of Document objects (with .page_content).
"""

def load_and_split(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # Loads entire PDF (one Document per page by default).
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)  # Split into smaller chunks.
    return chunks


