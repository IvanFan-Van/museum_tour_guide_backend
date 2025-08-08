# Data model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agent.prompts import RANKING_PROMPT
from agent.utils import zero_temp_llm as llm


class GradeDocument(BaseModel):
    """Rank retrieved text block relevance to a query."""

    reasoning: str = Field(
        description="Analysis of the block, identifying key information and how it relates to the query"
    )
    relevance_score: float = Field(
        description="Relevance score from 0 to 1, where 0 is Completely Irrelevant and 1 is Perfectly Relevant"
    )


class GradeDocuments(BaseModel):
    """Rank retrieved multiple text blocks relevance to a query."""

    block_rankings: list[GradeDocument] = Field(
        description="A list of text blocks and their associated relevance scores."
    )


# LLM with function call
structured_llm_grader = llm.with_structured_output(GradeDocuments)


grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RANKING_PROMPT),
        ("human", "{user_prompt}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
