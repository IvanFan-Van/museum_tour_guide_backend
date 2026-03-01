"""
Agent 抽象层

该模块定义了与具体 Agent 实现（如 LangGraph）解耦的统一接口，
使上层（WebSocket、HTTP 等表示层）无需感知底层实现细节。

层次结构：
    BaseAgent          —— 抽象接口，定义 stream() 合约
    └── LangGraphAgent —— 基于 LangGraph CompiledGraph 的具体实现
"""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Literal, Sequence

from langchain_core.runnables.schema import StreamEvent
from pydantic import BaseModel

from src.ws_schema import (
    ArtifactResultPayload,
    StatusPayload,
    TextChunkPayload,
    WSArtifactMessage,
    WSMessage,
    WSStatusMessage,
    WSTextChunkMessage,
)
from src.utils import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# 输入模型
# ---------------------------------------------------------------------------


class AgentInput(BaseModel):
    """Agent 的统一入参模型，屏蔽传输层的原始数据格式。"""

    query: str
    language: Literal["zh", "en"] = "zh"
    section_idx: str | None = None


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """
    Agent 抽象基类。

    所有 Agent 实现必须继承此类并实现 `stream` 方法。
    该方法是一个异步生成器，接收标准化的 `AgentInput`，
    逐条 yield `WSMessage`，供上层传输层直接发送给客户端。
    """

    @abstractmethod
    def stream(self, agent_input: AgentInput) -> AsyncGenerator[WSMessage, None]:
        """
        处理用户查询并以流式方式返回消息。

        Args:
            agent_input: 标准化的查询入参。

        Yields:
            WSMessage — 可以是文本块、状态消息或产物消息，
                        由上层传输层负责序列化并发送。

        Raises:
            Exception: 如果 Agent 执行过程中出现不可恢复的错误。
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# LangGraph 实现
# ---------------------------------------------------------------------------


class LangGraphAgent(BaseAgent):
    """
    基于 LangGraph CompiledGraph 的 Agent 实现。

    内部通过 `graph.astream_events` 驱动执行，将 LangGraph 事件
    转换为 `WSMessage` 类型后 yield 给调用方。

    事件处理策略
    ────────────
    - on_chat_model_stream  → WSTextChunkMessage(is_final=False)  实时文本流
    - on_chat_model_end     → WSTextChunkMessage(is_final=True)   文本流结束
    - on_chain_start[rag]   → WSStatusMessage(status="retrieving") 检索状态
    - graph 执行完成        → WSArtifactMessage                    图像 & 引用
    - 任意异常              → WSStatusMessage(status="error")      错误信息
    """

    def __init__(self, graph) -> None:
        """
        Args:
            graph: 已编译的 LangGraph CompiledGraph 实例。
        """
        self._graph = graph

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    def _build_graph_input(self, agent_input: AgentInput) -> dict:
        """将 AgentInput 转换为 LangGraph graph 的输入字典。"""
        return {
            "messages": [{"role": "user", "content": agent_input.query}],
            "language": agent_input.language,
            "section_idx": agent_input.section_idx,
        }

    def _process_event(
        self,
        event: StreamEvent,
        agent_run_ids: set[str],
    ) -> WSMessage | None:
        """
        将单个 LangGraph 流式事件转换为零或一条 WSMessage。

        Args:
            event:         单个 astream_events 事件。
            agent_run_ids: 持久化的可变集合，记录每次 agent 节点调用的 run_id。
                           由调用方在多次调用间共享，确保 ReAct 循环中每轮 LLM
                           调用均能正确过滤。

        Returns:
            需要向客户端推送的消息，或 None 表示该事件无需处理。
        """
        event_type: str = event["event"]
        event_name: str = event["name"]
        run_id: str = event.get("run_id", "")
        parent_ids: Sequence[str] = event.get("parent_ids", [])

        # 记录每次 agent 节点启动时的 run_id，支持 ReAct 循环中多轮调用
        if event_type == "on_chain_start" and event_name == "agent":
            agent_run_ids.add(run_id)
            return None

        # 实时文本块：来自任意 agent 节点内部的 LLM 流式输出
        if event_type == "on_chat_model_stream" and agent_run_ids.intersection(
            parent_ids
        ):
            chunk = event.get("data", {}).get("chunk")
            if chunk is None:
                logger.warning(f"Missing chunk in event: {event}")
                return None
            content: str = chunk.content
            if content:
                return WSTextChunkMessage(
                    payload=TextChunkPayload(content=content, is_final=False)
                )
            return None

        # 文本流结束标志：仅在 LLM 调用完成、属于 agent 节点且确实产生了文本时发送
        # （纯工具调用的轮次 output.content 为空，不应发送 is_final 标志）
        if event_type == "on_chat_model_end" and agent_run_ids.intersection(parent_ids):
            output = event.get("data", {}).get("output")
            if output and output.content:
                return WSTextChunkMessage(
                    payload=TextChunkPayload(content="", is_final=True)
                )
            return None

        # 检索状态提示
        if event_type == "on_chain_start" and event_name in ("rag", "retrieve_section"):
            return WSStatusMessage(
                payload=StatusPayload(
                    status="retrieving",
                    detail="Retrieving relevant documents...",
                )
            )

        # Web 搜索状态提示
        if event_type == "on_chain_start" and event_name == "web_search":
            return WSStatusMessage(
                payload=StatusPayload(
                    status="searching",
                    detail="Searching the web for more information...",
                )
            )

        return None

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    async def stream(self, agent_input: AgentInput) -> AsyncGenerator[WSMessage, None]:  # type: ignore[override]
        """
        执行 LangGraph graph 并以流式方式 yield WSMessage。

        流程：
        1. 通过 astream_events 驱动 graph 运行；
        2. 将每个事件转换为对应的 WSMessage 并 yield；
        3. graph 执行完成后，提取最终状态（images、references）并
           yield 一条 WSArtifactMessage；
        4. 若执行过程中抛出异常，yield 一条错误 WSStatusMessage 后终止。
        """
        graph_input = self._build_graph_input(agent_input)
        agent_run_ids: set[str] = set()  # 跟踪所有 agent 节点调用的 run_id
        final_state: dict = {}

        try:
            async for event in self._graph.astream_events(graph_input, version="v2"):
                message = self._process_event(event, agent_run_ids)
                if message is not None:
                    yield message

                # 捕获最终输出状态（on_chain_end 携带 graph 级别输出）
                if event["event"] == "on_chain_end" and event["name"] == "LangGraph":
                    final_state = event.get("data", {}).get("output", {})

        except Exception as exc:
            logger.error(
                f"LangGraphAgent encountered an error: {exc}\n{traceback.format_exc()}"
            )
            yield WSStatusMessage(
                payload=StatusPayload(
                    status="error",
                    detail=str(exc),
                )
            )
            return

        # ── 发送最终产物（图像 & 参考链接）─────────────────────────
        images: list[str] = final_state.get("retrieved_images") or []
        references: list[str] = final_state.get("references") or []

        if images or references:
            yield WSArtifactMessage(
                payload=ArtifactResultPayload(
                    images=images,
                    references=references,
                )
            )
