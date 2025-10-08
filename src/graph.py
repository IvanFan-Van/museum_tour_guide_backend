from langgraph.graph import StateGraph, END, START
from src.retrieval_graph import retrieval_graph
from src.models import State
from src.nodes import router, generator
from src.edges import to_retrieval


"""创建并返回 LangGraph"""
workflow = StateGraph(State)

# 添加节点
workflow.add_node("generator", generator)
workflow.add_node("router", router)
workflow.add_node("retrieval", retrieval_graph)

# 添加边
workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    to_retrieval,
    {
        "rag": "retrieval",  # 如果需要RAG，则跳转到RAG节点
        "no_rag": "generator",  # 否则跳转到聊天机器人节点
    },
)
workflow.add_edge("retrieval", "generator")
workflow.add_edge("generator", END)


# 编译图
graph = workflow.compile()
