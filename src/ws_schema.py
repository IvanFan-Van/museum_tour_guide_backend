"""WebSocket 消息的 Pydantic 模型定义"""

from typing import Dict, Any, Literal, Union
from pydantic import BaseModel


# --- Payload 类型 ---


class QueryPayload(BaseModel):
    text: str
    images: list[dict[Literal["format", "data"], str]]


class TextChunkPayload(BaseModel):
    content: str
    is_final: bool


class ControlPayload(BaseModel):
    action: str


class StatusPayload(BaseModel):
    status: str
    detail: str


ArtifactPayload = Dict[str, Any]


# --- 消息类型 ---


class WSQueryMessage(BaseModel):
    type: Literal["query"] = "query"
    payload: QueryPayload


class WSTextChunkMessage(BaseModel):
    type: Literal["text_chunk"] = "text_chunk"
    payload: TextChunkPayload


class WSControlMessage(BaseModel):
    type: Literal["control"] = "control"
    payload: ControlPayload


class WSStatusMessage(BaseModel):
    type: Literal["status"] = "status"
    payload: StatusPayload


class WSArtifactMessage(BaseModel):
    type: Literal["artifact"] = "artifact"
    payload: ArtifactPayload


# --- 联合类型 ---

WSMessage = Union[
    WSQueryMessage,
    WSTextChunkMessage,
    WSControlMessage,
    WSStatusMessage,
    WSArtifactMessage,
]
