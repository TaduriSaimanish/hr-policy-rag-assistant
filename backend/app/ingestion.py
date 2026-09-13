import os, logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def process_pdf(file_path: str) -> List[Document]:
    """
    Extracts pages from a PDF document preserving page numbers and chunks text.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    page_info = [page.page_content for page in pages]
    for i in page_info:
        logger.info("process_pdf: page information after loading is %s",i)
    
    # Recursive character splitting (500-800 token target with 10-15% overlap)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(pages)
    
    # Standardize chunk metadata for attribution
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{os.path.basename(file_path)}_p{chunk.metadata.get('page', 0)+1}_c{idx}"
        chunk.metadata["page_label"] = chunk.metadata.get("page", 0) + 1
        chunk.metadata["source_file"] = os.path.basename(file_path)
        
    return chunks
