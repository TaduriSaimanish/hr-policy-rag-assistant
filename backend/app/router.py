from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.llm import get_llm
import logging

logger = logging.getLogger(__name__)
class QueryDecomposition(BaseModel):
    is_multipart: bool = Field(description="True if query contains multiple distinct sub-questions.")
    sub_queries: List[str] = Field(description="List of decomposed single-topic search queries.")

parser = PydanticOutputParser(pydantic_object=QueryDecomposition)

router_prompt = ChatPromptTemplate.from_template(
    """Analyze the following user query in the context of HR policy documents.
Determine if it contains multiple questions or requires fetching distinct pieces of information.
If it is multi-part, break it down into explicit single-topic sub-queries.
If single-topic, set is_multipart to false and return only the original query in sub_queries.

{format_instructions}

User Query: {query}
"""
).partial(format_instructions=parser.get_format_instructions())

def decompose_query(query: str) -> List[str]:
    """Decomposes complex user queries into sub-queries."""
    llm = get_llm()
    chain = router_prompt | llm | parser

    result: QueryDecomposition = chain.invoke({"query": query})
    return result.sub_queries if result.sub_queries else [query]

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert query optimiser for a document retrieval system. "
                "Your task is to rewrite the user's question to be more precise and "
                "retrieval-friendly, using technical language where appropriate. "
                "Return ONLY the rewritten query — no explanation, no preamble."
            ),
        ),
        ("human", "Original question: {question}"),
    ]
)

def optimise_query(question: str) -> str:
    """Stage 1: Rewrite the question for better vector-space retrieval."""
    llm = get_llm()
    try:
        chain = REWRITE_PROMPT | llm
        rewritten = chain.invoke({"question":question})
        logger.info("Query rewrite: '%s' → '%s'", question, rewritten.content)
        return rewritten.content.strip()
    except Exception as exc:
        logger.warning("Query rewrite failed (%s), using original question.", exc)
        return question