from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from app.llm import get_llm

SYSTEM_PROMPT = """You are an authoritative HR Policy Assistant. Answer the user's question using ONLY the provided context chunks.
If the requested information is not explicitly stated in the context, respond: "Information not found in the uploaded document."

Rules:
1. Do not use external knowledge or invent facts.
2. Base every claim on the context below.

Context:
{context}

Question: {question}
Answer:"""


rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful and knowledgeable AI learning assistant. "
                "Answer the user's question using ONLY the provided context. "
                "If the context does not contain enough information, say so honestly. "
                "Cite relevant page numbers when possible. "
                "Be thorough, structured, and educational in your response."
            ),
        ),
        (
            "human",
            (
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
        ),
    ]
)

prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

def format_docs(docs: List[Document]) -> str:
    formatted = []
    for doc in docs:
        source_info = f"[Page {doc.metadata.get('page_label')}, Chunk: {doc.metadata.get('chunk_id')}]"
        formatted.append(f"{source_info}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

def build_lcel_chain():
    llm = get_llm()

    chain = (
        {"context": lambda x: format_docs(x["documents"]), "question": lambda x: x["question"]}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return chain

def execute_rag(question: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
    chain = build_lcel_chain()
    answer = chain.invoke({"question": question, "documents": retrieved_docs})
    
    sources = [
        {
            "chunk_id": doc.metadata.get("chunk_id"),
            "page_label": doc.metadata.get("page_label"),
            "source_file": doc.metadata.get("source_file"),
            "snippet": doc.page_content
        }
        for doc in retrieved_docs
    ]
    
    return {"answer": answer, "sources": sources}
