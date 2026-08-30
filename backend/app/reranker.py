from typing import List, Tuple, Dict, Any
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder = CrossEncoder(RERANKER_MODEL)

def rerank_documents(
    query: str, documents: List[Document], top_n: int = 4
) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """
    Re-scores vector candidates via Cross-Encoder and logs before/after rank positions.
    """
    if not documents:
        return [], []
        
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)
    
    scored_docs = []
    comparison_log = []
    
    for idx, (doc, score) in enumerate(zip(documents, scores)):
        scored_docs.append({
            "doc": doc,
            "score": float(score),
            "pre_rank": idx + 1
        })
        
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    
    reranked_docs = []
    for post_rank, item in enumerate(scored_docs[:top_n], start=1):
        doc = item["doc"]
        reranked_docs.append(doc)
        comparison_log.append({
            "chunk_id": doc.metadata.get("chunk_id"),
            "page": doc.metadata.get("page_label"),
            "pre_rerank_position": item["pre_rank"],
            "post_rerank_position": post_rank,
            "rerank_score": round(item["score"], 4),
            "snippet": doc.page_content[:120] + "..."
        })
        
    return reranked_docs, comparison_log
