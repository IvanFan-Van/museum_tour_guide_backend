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
    tool_name: str | None
    tool_args: dict | None
    route_reason: str | None


class RouteDecision(BaseModel):
    should_call: bool = Field(description="Whether to call a tool for this query")
    tool_name: str | None = Field(
        default=None, description="Tool name to call when should_call is True"
    )
    arguments: dict | None = Field(
        default=None, description="Arguments for the tool call"
    )
    reason: str = Field(description="Briefly explain the criteria for judgment")
