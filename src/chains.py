"""定义 LangChain 中的 Runnable Chain 对象"""

from src.utils import get_gpt4o
from src.models import QueryRouting
from src.prompts import QUERY_ROUTER_PROMPT, GENERATOR_PROMPT
from langchain_core.prompts import ChatPromptTemplate

gpt4o = get_gpt4o()

# QUERY ROUTER
structured_llm_grader = gpt4o.with_structured_output(QueryRouting)
query_router_prompt = ChatPromptTemplate(
    [("system", QUERY_ROUTER_PROMPT), ("human", "User Query: {query}")]
)
query_router = query_router_prompt | structured_llm_grader


# RAG GENERATOR
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GENERATOR_PROMPT),
        (
            "user",
            'Here are some relevant documents:\n{docs}\n\n"'
            '"Based on these, answer this query "{query}"',
        ),
    ]
)
rag_generator = prompt | gpt4o


# NON-RAG GENERATOR
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GENERATOR_PROMPT),
        ("user", "{query}"),
    ]
)
direct_generator = prompt | gpt4o
