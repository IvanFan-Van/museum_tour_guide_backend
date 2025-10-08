"""定义工作流中的边"""

from src.models import State


def to_retrieval(state: State):
    """条件函数 - 判断是否需要RAG"""
    return "rag" if state["need_rag"] else "no_rag"
