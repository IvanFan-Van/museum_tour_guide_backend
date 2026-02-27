"""主 RAG 工作流：包含 chains、节点函数和 graph 组装"""

from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate

from src.retrieval_graph import retrieval_graph
from src.models import State, QueryRouting
from src.prompts import QUERY_ROUTER_PROMPT, GENERATOR_PROMPT
from src.utils import get_gpt4o, format_docs, get_logger

logger = get_logger()


# --- CHAINS ---


def _build_chains():
    gpt4o = get_gpt4o()
    router_prompt = ChatPromptTemplate(
        [("system", QUERY_ROUTER_PROMPT), ("human", "User Query: {query}")]
    )
    query_router = router_prompt | gpt4o.with_structured_output(QueryRouting)
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATOR_PROMPT),
            (
                "user",
                "Here are some relevant documents:\n{docs}\n\n"
                'Based on these, answer this query "{query}"',
            ),
        ]
    )
    rag_generator = rag_prompt | gpt4o

    direct_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATOR_PROMPT),
            ("user", "{query}"),
        ]
    )
    direct_generator = direct_prompt | gpt4o

    return query_router, rag_generator, direct_generator


query_router, rag_generator, direct_generator = _build_chains()


# --- NODES ---


async def router(state: State):
    """判断是否需要 RAG 检索"""
    if state.get("doc_id"):
        logger.info(f"Doc ID provided: {state['doc_id']}, routing to RAG")
        return {"need_rag": True}
    response = await query_router.ainvoke({"query": state["messages"][-1].content})
    return {"need_rag": response.need_rag}  # type: ignore


async def generator(state: State):
    """根据检索结果（或无检索）生成回复"""
    if state.get("docs") and len(state["docs"]) > 0:
        response = await rag_generator.ainvoke(
            {"query": state["messages"][-1].content, "docs": format_docs(state["docs"])}
        )
    else:
        response = await direct_generator.ainvoke(
            {"query": state["messages"][-1].content}
        )
    return {"messages": [response]}


# --- EDGES ---


def to_retrieval(state: State):
    """条件边：判断是否需要 RAG"""
    return "rag" if state["need_rag"] else "no_rag"


# --- GRAPH ---

workflow = StateGraph(State)

workflow.add_node("router", router)
workflow.add_node("retrieval", retrieval_graph)
workflow.add_node("generator", generator)

workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    to_retrieval,
    {
        "rag": "retrieval",
        "no_rag": "generator",
    },
)
workflow.add_edge("retrieval", "generator")
workflow.add_edge("generator", END)

graph = workflow.compile()
