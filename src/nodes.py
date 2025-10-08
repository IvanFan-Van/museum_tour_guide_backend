"""定义工作流中的节点"""

from src.models import State
from src.chains import query_router, rag_generator, direct_generator
from src.utils import format_docs, get_logger

logger = get_logger()


# 定义 router 节点
async def router(state: State):
    if state.get("doc_id"):
        logger.info(f"Doc ID provided: {state['doc_id']}, routing to RAG")
        return {"need_rag": True}
    response = await query_router.ainvoke(state["messages"][-1].content)
    return {"need_rag": response.need_rag}


# 定义聊天机器人节点
async def generator(state: State):
    """聊天机器人节点 - 处理用户消息并生成回复"""
    if state.get("docs") and len(state["docs"]) > 0:
        response = await rag_generator.ainvoke(
            {"query": state["messages"][-1].content, "docs": format_docs(state["docs"])}
        )
        return {"messages": [response]}
    else:
        response = await direct_generator.ainvoke(
            {"query": state["messages"][-1].content}
        )
        return {"messages": [response]}
