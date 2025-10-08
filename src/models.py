"""定义数据模型"""

from typing import Annotated, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    need_rag: bool
    docs: list[Document]
    doc_id: str | None  # QR Code 返回的文档 ID


class QueryRouting(BaseModel):
    need_rag: bool = Field(
        description="whether need rag, if need rag pipeline, set to True, otherwise False"
    )
    reason: str = Field(description="Briefly explain the criteria for judgment")
