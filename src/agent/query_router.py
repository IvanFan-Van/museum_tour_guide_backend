from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agent.prompts import QUERY_ROUTER_PROMPT
from agent.utils import llm


class QueryRouting(BaseModel):
    need_rag: bool = Field(description="whether need rag")
    reason: str = Field(description="Briefly explain the criteria for judgment")


structured_llm_grader = llm.with_structured_output(QueryRouting)

query_router_prompt = ChatPromptTemplate(
    [("system", QUERY_ROUTER_PROMPT), ("human", "User Query: {question}")]
)

query_router = query_router_prompt | structured_llm_grader
