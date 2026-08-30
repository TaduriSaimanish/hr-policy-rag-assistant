import json
from datetime import datetime
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

LOG_FILE = "ragas_eval_logs.jsonl"

def run_ragas_evaluation(question: str, answer: str, contexts: List[str], ground_truth: str = "") -> Dict[str, float]:
    """Calculates RAGAS metrics per query and appends to a JSONL log file."""
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    if ground_truth:
        data["ground_truth"] = [ground_truth]
        
    dataset = Dataset.from_dict(data)
    
    metrics = [faithfulness, answer_relevance]
    if ground_truth:
        metrics.append(context_precision)
        
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings()
    
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )
    
    scores = {metric.name: float(score) for metric, score in results.items()}
    
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "metrics": scores
    }
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    return scores
