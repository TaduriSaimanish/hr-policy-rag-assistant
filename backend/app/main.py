import os
import shutil
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.ingestion import process_pdf
from app.vectorstore import create_and_save_index, similarity_search
from app.router import decompose_query
from app.reranker import rerank_documents
from app.rag_chain import execute_rag
from app.evaluator import run_ragas_evaluation

app = FastAPI(title="HR Policy RAG Assistant API", version="1.0.0")

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
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    chunks = process_pdf(file_path)
    create_and_save_index(chunks)
    
    return {
        "message": "File successfully uploaded and indexed.",
        "filename": file.filename,
        "total_chunks": len(chunks)
    }

@app.post("/api/query")
def process_user_query(request: QueryRequest):
    query = request.question
    
    # Step 1: Query Decomposition & Routing
    sub_queries = decompose_query(query)
    
    # Step 2: Vector Retrieval across sub-queries
    raw_candidates = []
    seen_ids = set()
    for sq in sub_queries:
        docs = similarity_search(sq, k=6)
        for doc in docs:
            c_id = doc.metadata.get("chunk_id")
            if c_id not in seen_ids:
                seen_ids.add(c_id)
                raw_candidates.append(doc)
                
    # Step 3: Cross-Encoder Re-Ranking
    reranked_docs, rank_comparison = rerank_documents(query, raw_candidates, top_n=4)
    
    # Step 4: LCEL Chain Answer Generation
    rag_response = execute_rag(query, reranked_docs)
    
    # # Step 5: RAGAS Evaluation Logging
    # contexts = [doc.page_content for doc in reranked_docs]
    # ragas_scores = run_ragas_evaluation(
    #     question=query,
    #     answer=rag_response["answer"],
    #     contexts=contexts,
    #     ground_truth=request.ground_truth
    # )
    
    return {
        "question": query,
        "decomposed_queries": sub_queries,
        "answer": rag_response["answer"],
        "sources": rag_response["sources"],
        "rerank_comparison": rank_comparison
        # "ragas_scores": ragas_scores
    }
