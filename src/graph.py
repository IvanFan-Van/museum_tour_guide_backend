import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import json
import yaml
import requests
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from supabase import create_client, Client
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, END
from typing import Annotated, Any, Literal, Sequence, TypedDict, cast
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

load_dotenv(find_dotenv(), override=True)

# ── 加载配置文件 ────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

# ── SiliconFlow API 配置 ────────────────────────────────────
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = _cfg["siliconflow"]["base_url"]
EMBED_MODEL = _cfg["siliconflow"]["embed_model"]
RERANK_MODEL = _cfg["siliconflow"]["rerank_model"]
RERANK_INSTRUCTION = _cfg["siliconflow"]["rerank_instruction"]
_EMBED_TIMEOUT = _cfg["siliconflow"]["timeout"]["embed"]
_RERANK_TIMEOUT = _cfg["siliconflow"]["timeout"]["rerank"]

# ── 检索参数 ────────────────────────────────────────────────
_MATCH_COUNT = _cfg["retrieval"]["match_count"]
_RERANK_THRESHOLD = _cfg["retrieval"]["rerank_threshold"]
_RERANK_TOP_K = _cfg["retrieval"]["rerank_top_k"]

# ── Supabase 表 / RPC 名称 ──────────────────────────────────
_TBL_SECTIONS = _cfg["supabase"]["tables"]["sections"]
_TBL_IMAGES = _cfg["supabase"]["tables"]["images"]
_RPC_MATCH = _cfg["supabase"]["rpc"]["match_paragraphs"]

_sf_headers = {
    "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
    "Content-Type": "application/json",
}

supabase: Client = create_client(
    os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_KEY", "")
)


# ── State 定义 ──────────────────────────────────────────────
class AgentState(TypedDict):
    # ===== 输入参数 =====
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: Literal["zh", "en"]
    section_idx: str | None  # 指定要优先获取的 section，为 None 时走普通 RAG 流程

    # ===== 图状态 =====
    retrieved_docs: list[str]
    retrieved_page_idxs: list[
        int
    ]  # 与 retrieved_docs 平行，记录每篇文档对应的 page_idx
    reranked_docs: list[str]
    retrieved_images: list[str]
    references: list[str]  # web_search 返回的网页链接
    current_query: str
    tool_call_id: str  # 用于在 subgraph 内部传递 tool_call_id


# ── SiliconFlow API 封装 ───────────────────────────────────
def get_embedding(text: str) -> list[float]:
    """调用 SiliconFlow Embedding API，返回单条文本的向量。"""
    resp = requests.post(
        f"{SILICONFLOW_BASE_URL}/embeddings",
        headers=_sf_headers,
        json={"model": EMBED_MODEL, "input": text, "encoding_format": "float"},
        timeout=_EMBED_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def rerank_documents(
    query: str, documents: list[str], threshold: float = _RERANK_THRESHOLD
) -> list[str]:
    """调用 SiliconFlow Rerank API，返回相关性得分超过阈值的文档（按得分降序）。"""
    if not documents:
        return []
    resp = requests.post(
        f"{SILICONFLOW_BASE_URL}/rerank",
        headers=_sf_headers,
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "instruction": RERANK_INSTRUCTION,
            "return_documents": False,
        },
        timeout=_RERANK_TIMEOUT,
    )
    resp.raise_for_status()
    results = resp.json()["results"]  # [{"index": int, "relevance_score": float}, ...]
    scored = sorted(results, key=lambda r: r["relevance_score"], reverse=True)
    return [documents[r["index"]] for r in scored if r["relevance_score"] > threshold]


# ── Subgraph 节点定义 ───────────────────────────────────────
def retrieve_node(state: AgentState):
    """Subgraph 节点 1：向量召回，记录文档与 page_idx 的对应关系"""
    query = state["current_query"]
    language = state.get("language", "en")
    text_field = "zh_text" if language == "zh" else "en_text"

    embeddings = get_embedding(query)
    results = supabase.rpc(
        _RPC_MATCH, {"query_embedding": embeddings, "match_count": _MATCH_COUNT}
    ).execute()

    if not results.data:
        return {"retrieved_docs": [], "retrieved_page_idxs": []}

    rpc_data = cast(list[dict[str, Any]], results.data)
    section_idxs = list(set(item["section_idx"] for item in rpc_data))
    sections = (
        supabase.table(_TBL_SECTIONS)
        .select(f"{text_field}, page_idx")
        .in_("section_idx", section_idxs)
        .execute()
    )
    sections_data = cast(list[dict[str, Any]], sections.data)
    docs = [item[text_field] for item in sections_data]
    page_idxs = [item["page_idx"] for item in sections_data]

    return {"retrieved_docs": docs, "retrieved_page_idxs": page_idxs}


def rerank_node(state: AgentState):
    """Subgraph 节点 2：Rerank 精排，并将结果写回消息列表"""
    query = state["current_query"]
    docs = state.get("retrieved_docs", [])
    page_idxs = state.get("retrieved_page_idxs", [])
    call_id = state["tool_call_id"]

    final_docs = rerank_documents(query, docs)[:_RERANK_TOP_K]

    # 若精排后没有高置信度文档，返回明确的空结果信号，让 LLM 知道需要 fallback
    if final_docs:
        # 建立 doc -> page_idx 映射，取精排后文档对应的 page_idx 去重后查询图像
        doc_to_page = {doc: pid for doc, pid in zip(docs, page_idxs)}
        final_page_idxs = list(
            set(doc_to_page[doc] for doc in final_docs if doc in doc_to_page)
        )
        images_result = (
            supabase.table(_TBL_IMAGES)
            .select("image_path")
            .in_("page_idx", final_page_idxs)
            .execute()
        )
        images = [
            item["image_path"]
            for item in cast(list[dict[str, Any]], images_result.data)
        ]

        content = json.dumps(
            {"documents": final_docs, "images": images},
            ensure_ascii=False,
        )
    else:
        images = []
        content = json.dumps(
            {
                "result": "no_relevant_documents",
                "message": "The proprietary database did not return sufficiently relevant results for this query.",
            },
            ensure_ascii=False,
        )

    tool_message = ToolMessage(
        content=content,
        name="retrieve_documents",
        tool_call_id=call_id,
    )

    return {
        "reranked_docs": final_docs,
        "retrieved_images": images,
        "messages": [tool_message],
    }


# ── 构建 Subgraph ───────────────────────────────────────────
rag_subgraph_builder = StateGraph(AgentState)
rag_subgraph_builder.add_node("retrieve", retrieve_node)
rag_subgraph_builder.add_node("rerank", rerank_node)
rag_subgraph_builder.set_entry_point("retrieve")
rag_subgraph_builder.add_edge("retrieve", "rerank")
rag_subgraph_builder.set_finish_point("rerank")

rag_subgraph = rag_subgraph_builder.compile()  # 编译为可复用的 Runnable

# ── 工具定义 ────────────────────────────────────────────────
search = DuckDuckGoSearchResults(
    output_format="list", num_results=_cfg["web_search"]["num_results"]
)
llm = AzureChatOpenAI(
    model=_cfg["llm"]["model"], api_version=os.getenv("AZURE_OPENAI_API_VERSION", "")
)


@tool
def web_search(query: str):
    """Call to perform a web search using DuckDuckGo when the museum database has no relevant results."""
    pass


@tool
def retrieve_documents(query: str):
    """Call to retrieve relevant documents based on a query search in the museum database."""
    pass


@tool
def retrieve_section(section_idx: str):
    """Fetch the full content and related images for a specific section by its section_idx."""
    pass


tools = [retrieve_documents, retrieve_section, web_search]
model = llm.bind_tools(tools)


# ── web_search 节点 ────────────────────────────────────────────────────────
def web_search_node(state: AgentState):
    """执行网页搜索，提取链接保存至 references，并将结果以 ToolMessage 注入对话"""
    last_message = cast(AIMessage, state["messages"][-1])
    tool_call = last_message.tool_calls[0]
    query = tool_call["args"]["query"]
    call_id = tool_call["id"]

    results = search.invoke(query)

    new_references = [
        item["link"] for item in results if isinstance(item, dict) and "link" in item
    ]

    tool_message = ToolMessage(
        content=json.dumps(results, ensure_ascii=False),
        name="web_search",
        tool_call_id=call_id,
    )

    return {
        "references": (state.get("references") or []) + new_references,
        "messages": [tool_message],
    }


# ── retrieve_section 节点（直接按 section_idx 查询，不经过 embedding）──────
def retrieve_section_node(state: AgentState):
    """直接按 section_idx 查询文档内容及关联图像，以 ToolMessage 注入对话"""
    last_message = cast(AIMessage, state["messages"][-1])
    tool_call = last_message.tool_calls[0]
    section_idx_arg = tool_call["args"]["section_idx"]
    call_id = tool_call["id"]
    language = state.get("language", "zh")
    text_field = "zh_text" if language == "zh" else "en_text"

    section_result = (
        supabase.table(_TBL_SECTIONS)
        .select(f"{text_field}, page_idx")
        .eq("section_idx", section_idx_arg)
        .execute()
    )

    if not section_result.data:
        content = json.dumps(
            {
                "result": "section_not_found",
                "message": f"No section found for section_idx: {section_idx_arg}",
            },
            ensure_ascii=False,
        )
        images = []
    else:
        row = cast(dict[str, Any], section_result.data[0])
        doc_text = row[text_field]
        page_idx = row["page_idx"]

        images_result = (
            supabase.table(_TBL_IMAGES)
            .select("image_path")
            .eq("page_idx", page_idx)
            .execute()
        )
        images = [
            item["image_path"]
            for item in cast(list[dict[str, Any]], images_result.data)
        ]
        content = json.dumps(
            {"documents": [doc_text], "images": images},
            ensure_ascii=False,
        )

    tool_message = ToolMessage(
        content=content,
        name="retrieve_section",
        tool_call_id=call_id,
    )
    return {
        "retrieved_images": images,
        "messages": [tool_message],
    }


# ── 主图节点定义 ────────────────────────────────────────────
def call_model(state: AgentState, config: RunnableConfig):
    language = state.get("language", "zh")
    section_idx = state.get("section_idx")

    if language == "zh":
        section_hint = (
            f"\n\n【当前请求上下文】本次对话指定了 section_idx='{section_idx}'。"
            "如果对话历史中尚未出现针对该 section_idx 的 'retrieve_section' 调用结果，"
            "则必须首先调用 'retrieve_section' 工具获取该 section 的内容，再决定是否需要进一步查询。"
            if section_idx
            else ""
        )
        system_content = (
            "你是一位博物馆导览员，请遵循以下工具使用策略：\n"
            "1. 如果当前上下文指定了 section_idx，且对话历史中尚未检索该 section，"
            "则必须首先调用 'retrieve_section' 工具获取对应内容。\n"
            "2. 对于涉及展品、藏品或博物馆历史的问题，如果对话历史中尚未包含相关检索结果，"
            "则调用 'retrieve_documents' 工具查询专有数据库。\n"
            "3. 如果对话历史中已存在足够的相关信息（来自 'retrieve_section' 或 'retrieve_documents'），"
            "则直接利用这些信息作答，无需重复调用工具。\n"
            "4. 如果 'retrieve_documents' 返回的结果为空或明确标注 'no_relevant_documents'，"
            "则改为调用 'web_search' 工具进行补充搜索。\n"
            "5. 自主判断当前上下文是否足够回答问题；如果不够，继续调用相应工具直到信息充足。\n"
            "6. 请用中文回答，回答应简短、精炼，并保持导览员的语气。" + section_hint
        )
    else:
        section_hint = (
            f"\n\n[Current request context] This conversation has section_idx='{section_idx}' specified. "
            "If the conversation history does not yet contain a 'retrieve_section' result for this section_idx, "
            "you must call 'retrieve_section' first before proceeding."
            if section_idx
            else ""
        )
        system_content = (
            "You are a museum tour guide. Follow this tool-use strategy:\n"
            "1. If a section_idx is specified in the current context and has not yet been retrieved in the "
            "conversation history, you must call 'retrieve_section' first to fetch its content.\n"
            "2. For questions about exhibits, collections, or museum history, call "
            "'retrieve_documents' to query the proprietary database — but only if "
            "relevant retrieved documents are not already present in the conversation history.\n"
            "3. If the conversation history already contains sufficient relevant information "
            "(from 'retrieve_section' or 'retrieve_documents'), use it directly without calling tools again.\n"
            "4. If 'retrieve_documents' returns an empty list or a 'no_relevant_documents' result, "
            "fall back to 'web_search' to supplement the information.\n"
            "5. Autonomously judge whether you have enough context to answer; if not, keep querying until you do.\n"
            "6. Answer in English. Keep answers short, concise and in the tone of a tour guide."
            + section_hint
        )
    system_prompt = SystemMessage(content=system_content)
    response = model.invoke([system_prompt] + list(state["messages"]), config)
    return {"messages": [response]}


def tool_node(state: AgentState):
    """Generic fallback for any other tools not handled by dedicated nodes."""
    last_message = cast(AIMessage, state["messages"][-1])
    outputs = []
    for tool_call in last_message.tool_calls:
        # 其他未定义专用节点的工具暂时返回空结果
        outputs.append(
            ToolMessage(
                content=json.dumps(
                    {"error": f"No dedicated node for tool '{tool_call['name']}'"},
                    ensure_ascii=False,
                ),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


def rag_entry_node(state: AgentState):
    """
    RAG 入口节点：从 tool_call 中提取 query 和 tool_call_id，
    写入状态后交给 subgraph 处理。
    subgraph 以相同的 AgentState schema 运行，状态自动合并回父图。
    """
    last_message = cast(AIMessage, state["messages"][-1])
    tool_call = last_message.tool_calls[0]
    query = tool_call["args"]["query"]
    call_id = tool_call["id"]

    # 调用 subgraph，传入预处理好的状态
    result = rag_subgraph.invoke(
        cast(
            AgentState,
            {
                **state,
                "current_query": query,
                "tool_call_id": call_id,
            },
        )
    )

    # 将 subgraph 产生的增量状态返回给父图合并
    return {
        "retrieved_docs": result["retrieved_docs"],
        "retrieved_page_idxs": result["retrieved_page_idxs"],
        "reranked_docs": result["reranked_docs"],
        "retrieved_images": result["retrieved_images"],
        "messages": result["messages"],
    }


# ── 路由函数 ────────────────────────────────────────────────
def should_continue(state: AgentState):
    last_message = cast(AIMessage, state["messages"][-1])
    if not last_message.tool_calls:
        return "end"
    tool_name = last_message.tool_calls[0]["name"]
    if tool_name == "retrieve_documents":
        return "rag"
    if tool_name == "retrieve_section":
        return "retrieve_section"
    if tool_name == "web_search":
        return "web_search"
    return "tools"


# ── 构建主图 ────────────────────────────────────────────────
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("rag", rag_entry_node)  # subgraph 作为单一节点挂载
workflow.add_node("retrieve_section", retrieve_section_node)
workflow.add_node("web_search", web_search_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "rag": "rag",
        "tools": "tools",
        "retrieve_section": "retrieve_section",
        "web_search": "web_search",
        "end": END,
    },
)

workflow.add_edge("rag", "agent")
workflow.add_edge("tools", "agent")
workflow.add_edge("retrieve_section", "agent")
workflow.add_edge("web_search", "agent")

graph = workflow.compile()
