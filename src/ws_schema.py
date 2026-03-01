"""WebSocket 消息的 Pydantic 模型定义"""

from typing import Dict, Any, Literal, Union
from pydantic import BaseModel


# --- Payload 类型 ---


# --- Payload 类型 ---


class QueryPayload(BaseModel):
    text: str
    images: list[dict[Literal["format", "data"], str]]
    section_idx: int | None = None
    language: Literal["en", "zh"] = "en"


class TextChunkPayload(BaseModel):
    content: str
    is_final: bool


class ControlPayload(BaseModel):
    action: str


class StatusPayload(BaseModel):
    status: str
    detail: str


ArtifactPayload = Dict[str, Any]


class ArtifactResultPayload(BaseModel):
    """Graph 执行完成后返回的结构化产物，包括图像和参考链接"""

    images: list[str] = []
    references: list[str] = []


# --- 消息类型 ---


class WSQueryMessage(BaseModel):
    type: Literal["query"] = "query"
    payload: QueryPayload


class WSTextChunkMessage(BaseModel):
    type: Literal["text_chunk"] = "text_chunk"
    payload: TextChunkPayload


# class WSControlMessage(BaseModel):
#     type: Literal["control"] = "control"
#     payload: ControlPayload


class WSStatusMessage(BaseModel):
    type: Literal["status"] = "status"
    payload: StatusPayload


class WSArtifactMessage(BaseModel):
    """Graph 执行结束后一次性发送的产物消息（图像路径、参考链接）"""

    type: Literal["artifact"] = "artifact"
    payload: ArtifactResultPayload


WSTextMessage = Union[
    WSQueryMessage,
    WSTextChunkMessage,
    WSStatusMessage,
    WSArtifactMessage,
    # WSControlMessage
]
WSByteMessage = bytes

# --- 联合类型 ---

WSMessage = Union[WSTextMessage, WSByteMessage]
