from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class QueryDecomposition(BaseModel):
    is_multipart: bool = Field(description="True if query contains multiple distinct sub-questions.")
    sub_queries: List[str] = Field(description="List of decomposed single-topic search queries.")

router_prompt = ChatPromptTemplate.from_template(
    """Analyze the following user query in the context of HR policy documents.
Determine if it contains multiple questions or requires fetching distinct pieces of information.
If it is multi-part, break it down into explicit single-topic sub-queries.
If single-topic, set is_multipart to false and return only the original query in sub_queries.

User Query: {query}
"""
)

def decompose_query(query: str, llm_model: str = "gpt-4o-mini") -> List[str]:
    """Decomposes complex user queries into sub-queries."""
    llm = ChatOpenAI(model=llm_model, temperature=0)
    structured_llm = llm.with_structured_output(QueryDecomposition)
    chain = router_prompt | structured_llm
    
    result: QueryDecomposition = chain.invoke({"query": query})
    return result.sub_queries if result.sub_queries else [query]
