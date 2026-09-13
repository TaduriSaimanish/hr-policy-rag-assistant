import os
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True})

def create_and_save_index(chunks: List[Document]) -> str:
    """Creates a local FAISS index with normalized vectors and persists it to disk."""
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    return INDEX_DIR

def load_or_get_index() -> FAISS:
    """Loads existing FAISS vector store from disk."""
    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError("FAISS index directory does not exist. Upload a PDF first.")
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def similarity_search(query: str, k: int = 10) -> List[Document]:
    """Queries vector index for initial top-K chunk retrieval."""
    vectorstore = load_or_get_index()
    return vectorstore.similarity_search(query, k=k)
