"""定义检索图子图 - 包含检索和重排序节点"""

import asyncio
from langgraph.graph import START, StateGraph, END
from langchain_core.documents import Document
from chromadb import PersistentClient
from src.models import State
from src.utils import get_logger
import requests
import os

logger = get_logger()
workflow = StateGraph(State)

# 全局客户端，避免重复初始化
_client = None
_collection = None

FINAL_DOCS_COUNT = 3  # 最终用于生成回答的文档数量
CLIENT_PATH = "./chroma_db"
COLLECTION_NAME = "museum_knowledge_base"


async def _init_chroma():
    """初始化 ChromaDB 客户端和集合"""
    global _client, _collection
    if _client is None:

        def _init_sync():
            try:
                client = PersistentClient(path=CLIENT_PATH)
                collection = client.get_collection(name=COLLECTION_NAME)
                return client, collection
            except Exception as e:
                raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

        _client, _collection = await asyncio.to_thread(_init_sync)

    return _client, _collection


async def _retrieve_documents(query: str) -> list[Document]:
    """异步检索文档"""
    _, collection = await _init_chroma()

    def _query_sync():
        results = collection.query(
            query_texts=[query],
            n_results=10,
            include=["documents", "metadatas"],
        )

        return [
            Document(
                id=id,
                page_content=doc,
                metadata=metadata,
            )
            for id, doc, metadata in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0]
            )
        ]

    return await asyncio.to_thread(_query_sync)


async def _retrieve_by_id(doc_id: str) -> list[Document]:
    _, collection = await _init_chroma()

    def _get_by_id_sync():
        try:
            results = collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"],
            )

            if not results["ids"] or len(results["ids"]) == 0:
                logger.info(f"Document with ID {doc_id} not found.")
                return []

            return [
                Document(
                    id=results["ids"][0],
                    page_content=results["documents"][0],
                    metadata=results["metadatas"][0],
                )
            ]

        except Exception as e:
            logger.info(f"Error retrieving document by ID {doc_id}: {e}")
            return []

    return await asyncio.to_thread(_get_by_id_sync)


async def _rerank_documents(query: str, docs: list[Document]) -> list[Document]:
    def _rerank_documents_sync():
        docs_raw = [doc.page_content for doc in docs]
        response = requests.post(
            os.environ["SILICONFLOW_RERANK_ENDPOINT"],
            headers={
                "Authorization": f"Bearer {os.environ['SILICONFLOW_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={
                "model": "BAAI/bge-reranker-v2-m3",
                "query": query,
                "documents": docs_raw,
            },
        )
        results = response.json()
        if "results" not in results:
            raise ValueError(f"Rerank API error: {results}")

        # 添加 rarank 分数
        for doc, res in zip(docs, results["results"]):
            doc.metadata["rerank_score"] = res["relevance_score"]

        # 根据分数排序
        ranked_docs = sorted(
            docs, key=lambda d: d.metadata["rerank_score"], reverse=True
        )

        ranked_docs = ranked_docs[:FINAL_DOCS_COUNT]  # 只保留前5个
        return ranked_docs

    return await asyncio.to_thread(_rerank_documents_sync)


async def retrieve(state: State):
    """检索节点 - 根据用户消息检索相关文档"""
    # 如果 state 中有 doc_id，直接根据 ID 检索
    if state.get("doc_id", None) and state["doc_id"] != "null":
        docs = await _retrieve_by_id(state["doc_id"])
        return {"docs": docs}

    # 否则根据消息内容检索
    query = state["messages"][-1].content
    docs = await _retrieve_documents(query)
    # TODO change "name" to "source"
    logger.info(
        f"Retrieved {len(docs)} documents for query: {query}: \n"
        + "\n".join(
            [
                f"\t\t{idx + 1}. {doc.metadata.get('name', 'unknown')}"
                for idx, doc in enumerate(docs)
            ]
        )
    )

    return {"docs": docs}


async def rerank(state: State):
    """重排序节点 - 对检索到的文档进行重排序"""
    docs = state.get("docs", [])
    if not docs:
        return {"docs": []}

    if state.get("doc_id") and len(docs) == 1:
        logger.info(
            f"Single document retrieved by ID {state['doc_id']}, skipping rerank."
        )
        return {"docs": docs}

    query = state["messages"][-1].content
    docs = state.get("docs", [])
    if not docs:
        return {"docs": []}
    ranked_docs = await _rerank_documents(query, docs)

    # expected logging: Reranked documents for query - "What is the history of...":
    #      - doc1 (score: 0.95)
    #      - doc2 (score: 0.89)
    logger.info(
        f'Reranked documents for query - "{query}": \n'
        + "\n".join(
            [
                f"\t\t{idx + 1}. {doc.metadata.get('name', 'unknown')}\t(score: {doc.metadata.get('rerank_score', 'N/A'):.4f})"
                for idx, doc in enumerate(ranked_docs)
            ]
        )
    )
    return {"docs": ranked_docs}


workflow.add_node("retrieve", retrieve)
workflow.add_node("rerank", rerank)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", END)
retrieval_graph = workflow.compile()
