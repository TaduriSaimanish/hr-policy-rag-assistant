import os,sys
import shutil
from typing import List, Optional
from dotenv import load_dotenv
import logging

load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.ingestion import process_pdf
from app.vectorstore import create_and_save_index, similarity_search
from app.router import decompose_query, optimise_query
from app.reranker import rerank_documents
from app.rag_chain import execute_rag
from app.evaluator import run_ragas_evaluation

app = FastAPI(title="HR Policy RAG Assistant API", version="1.0.0")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, "INFO", logging.INFO),
    stream=sys.stdout,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    question: str
    ground_truth: Optional[str] = ""

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "HR Policy RAG Assistant"}

@app.post("/api/upload")
def upload_pdf(file: UploadFile = File(...)):
    logger.info("upload_pdf: Ingesting the pdf has started")
    if not file.filename.endswith(".pdf"):
        logger.error("upload_pdf: Not a PDF file")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
   
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    logger.info("upload_pdf: file uploading is complete at %s",file_path)   
    chunks = process_pdf(file_path)
    logger.info("upload_pdf: created %d chunks from %s", len(chunks), file_path)
    for chunk in chunks:
        logger.info(
            "upload_pdf: chunk_id=%s page=%s chars=%d preview=%r",
            chunk.metadata.get("chunk_id"),
            chunk.metadata.get("page_label"),
            len(chunk.page_content),
            chunk.page_content[:80],
        )
    result = create_and_save_index(chunks)
    logger.info("Saved to vectore store database at %s",result)
    
    return {
        "message": "File successfully uploaded and indexed.",
        "filename": file.filename,
        "total_chunks": len(chunks)
    }

@app.post("/api/query")
def process_user_query(request: QueryRequest):
    query = request.question
    
    # Step 1: Query Optimisation
    optimised_query = optimise_query(query)
    logger.info("process_user_query")
    
    # Step 2: Vector Retrieval
    raw_candidates = similarity_search(optimised_query,25)
    for candidate in raw_candidates:
     logger.info("process_user_query: the retrieved raw_candidates are %s and metadata is %s",candidate.page_content,candidate.metadata)
                
    # Step 3: Cross-Encoder Re-Ranking
    reranked_docs, rank_comparison = rerank_documents(optimised_query, raw_candidates, top_n=5)

    
    # Step 4: LCEL Chain Answer Generation
    rag_response = execute_rag(query, reranked_docs)

    return {
        "question": query,
        "optimised_query": optimised_query,
        "answer": rag_response["answer"],
        "sources": rag_response["sources"],
        "rerank_comparison": rank_comparison
    }

class EvaluationRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = ""

@app.post("/api/evaluate")
def evaluate_response(request: EvaluationRequest):
    """Runs RAGAS scoring separately from the query flow so /api/query stays fast."""
    ragas_scores = run_ragas_evaluation(
        question=request.question,
        answer=request.answer,
        contexts=request.contexts,
        ground_truth=request.ground_truth
    )
    return {"ragas_scores": ragas_scores}
