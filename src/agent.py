"""
Agent 抽象层

该模块定义了与具体 Agent 实现解耦的统一接口，
使上层（WebSocket、HTTP 等表示层）无需感知底层实现细节。

层次结构：
    BaseAgent                —— 抽象接口，定义 stream() 合约
    └── LangGraphRemoteAgent —— 通过 langgraph-sdk 调用 LangGraph 部署 API 的实现
"""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Literal, Sequence

from pydantic import BaseModel

from src.models import (
    ArtifactPayload,
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
    thread_id: str
    language: Literal["zh", "en"] = "en"
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
# LangGraph Remote API 实现
# ---------------------------------------------------------------------------


class LangGraphRemoteAgent(BaseAgent):
    """
    通过 langgraph-sdk 调用 LangGraph 部署（Docker 容器）API 的 Agent 实现。

    内部通过 `client.runs.stream` 以 events + values 双模式订阅，将
    LangGraph 事件转换为 `WSMessage` 类型后 yield 给调用方。

    事件处理策略
    ────────────
    - stream_mode="events" / on_chat_model_stream  → WSTextChunkMessage(is_final=False)
    - stream_mode="events" / on_chat_model_end     → WSTextChunkMessage(is_final=True)
    - stream_mode="events" / on_chain_start[rag]   → WSStatusMessage(status="retrieving")
    - stream_mode="events" / on_chain_start[web]   → WSStatusMessage(status="searching")
    - stream_mode="values" (最后一条)              → 提取 images & references
    - graph 执行完成                               → WSArtifactMessage
    - 任意异常                                     → WSStatusMessage(status="error")
    """

    def __init__(self, url: str, graph_name: str = "graph") -> None:
        """
        Args:
            url:        LangGraph 部署 API 的根地址（对应 .env 中的 LANGGRAPH_URL）。
            graph_name: langgraph.json 中注册的图名称，用作 assistant_id。
        """
        from langgraph_sdk import get_client

        self._client = get_client(url=url)
        self._graph_name = graph_name

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
        event: dict,
        agent_run_ids: set[str],
    ) -> WSMessage | None:
        """
        将单个 LangGraph 流式事件转换为零或一条 WSMessage。

        Args:
            event:         stream_mode="events" 推送的单条事件字典，结构与
                           graph.astream_events 返回的事件一致。
            agent_run_ids: 持久化的可变集合，记录每次 agent 节点调用的 run_id。
                           由调用方在多次调用间共享，确保 ReAct 循环中每轮 LLM
                           调用均能正确过滤。

        Returns:
            需要向客户端推送的消息，或 None 表示该事件无需处理。
        """
        event_type: str = event.get("event", "")
        event_name: str = event.get("name", "")
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
            content: str = (
                chunk.get("content", "")
                if isinstance(chunk, dict)
                else getattr(chunk, "content", "")
            )
            if content:
                return WSTextChunkMessage(
                    payload=TextChunkPayload(content=content, is_final=False)
                )
            return None

        # 文本流结束标志：仅在 LLM 调用完成、属于 agent 节点且确实产生了文本时发送
        # （纯工具调用的轮次 output.content 为空，不应发送 is_final 标志）
        if event_type == "on_chat_model_end" and agent_run_ids.intersection(parent_ids):
            output = event.get("data", {}).get("output")
            if output is not None:
                content = (
                    output.get("content", "")
                    if isinstance(output, dict)
                    else getattr(output, "content", "")
                )
                if content:
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
        通过 LangGraph 远程 API 执行 graph 并以流式方式 yield WSMessage。

        流程：
        1. 确保 thread 存在（幂等创建）；
        2. 以 stream_mode=["events", "values"] 订阅 run；
        3. events 分片经 _process_event 转为 WSMessage 逐条 yield；
        4. values 分片持续更新 final_state，用于最终产物提取；
        5. 所有事件处理完毕后，yield 一条 WSArtifactMessage（若有产物）；
        6. 若执行过程中抛出异常，yield 一条错误 WSStatusMessage 后终止。
        """
        graph_input = self._build_graph_input(agent_input)
        agent_run_ids: set[str] = set()  # 跟踪所有 agent 节点调用的 run_id
        final_state: dict = {}

        # 幂等创建 thread：若已存在则复用，支持多轮对话历史
        await self._client.threads.create(
            thread_id=agent_input.thread_id,
            if_exists="do_nothing",
        )

        try:
            async for chunk in self._client.runs.stream(
                agent_input.thread_id,
                self._graph_name,
                input=graph_input,
                stream_mode=["events", "values"],
            ):
                # values 分片：持续记录最新图状态，用于结束时提取产物
                if chunk.event == "values":
                    final_state = chunk.data
                    continue

                # events 分片：转发给 _process_event 处理
                if chunk.event == "events":
                    message = self._process_event(chunk.data, agent_run_ids)
                    if message is not None:
                        yield message

                if chunk.event == "error":
                    error_detail = f"{chunk.data.get('error', 'Unknown error')}: {chunk.data.get('message')}"
                    raise Exception(f"LangGraph execution error: {error_detail}")

        except Exception as exc:
            logger.error(
                f"LangGraphRemoteAgent encountered an error: {exc}\n{traceback.format_exc()}"
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
                payload=ArtifactPayload(
                    images=images,
                    references=references,
                )
            )


# 向后兼容别名
LangGraphAgent = LangGraphRemoteAgent
