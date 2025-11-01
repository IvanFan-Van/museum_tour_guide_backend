import asyncio
import os
import traceback
from typing import TypedDict
import warnings
import json
from langchain_core.runnables.schema import StreamEvent
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.graph import graph
from src.utils import get_tts, get_logger
from src.accumulator import AudioAccumulator
from api_exception import register_exception_handlers

os.environ["ANONYMIZED_TELEMETRY"] = "False"

warnings.filterwarnings("ignore")
logger = get_logger()
origins = ["*", "http://localhost:5174"]

app = FastAPI(
    title="Museum Tour Guide API",
    description="API for the Museum Tour Guide application using RAG with LangGraph and LangChain",
    version="1.0.0",
)

register_exception_handlers(app, log_traceback=False, log=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/v1/error")
async def error():
    raise ValueError("This is a test error endpoint.")


@app.websocket("/api/v1/invoke")
async def invoke(websocket: WebSocket):
    await websocket.accept()
    acc = None

    # 获取节点ID的辅助函数
    def get_node_id(node_name: str, event):
        if event["name"] == node_name:
            return event["run_id"]
        else:
            return None

    try:
        data = await websocket.receive_json()
        query = data.get("query", None)
        doc_id = data.get("doc_id", None)

        if not query:
            raise ValueError("Query parameter is required.")

        graph_input = {
            "messages": [{"role": "user", "content": query}],
            "doc_id": doc_id,
        }

        # 结果队列, 存储任务完成后的结果
        queue = asyncio.Queue()

        tts = get_tts()
        acc = AudioAccumulator(tts_function=tts, num_sentence_cached=1)

        await websocket.send_json({"event": "connected", "data": {"status": "success"}})

        # 文本生成函数
        async def text_generation_task():
            event: StreamEvent
            generator_id = None
            async for event in graph.astream_events(graph_input, version="v2"):
                if generator_id is None:
                    generator_id = get_node_id("generator", event)

                if (
                    event["event"] == "on_chat_model_stream"
                    and generator_id in event["parent_ids"]
                ):
                    if "data" not in event or "chunk" not in event["data"]:
                        logger.error(f"No data in event: {event}")
                        continue

                    chunk = event["data"]["chunk"].content
                    if chunk:
                        data = {
                            "event": "message",
                            "data": {"chunk": chunk},
                        }
                        # 将结果添加到结果队列以及 accumulator 中
                        await queue.put(data)
                        await acc.add_chunk(chunk)
            await acc.flush()

        # 音频生成任务
        async def audio_generation_task():
            async for audio_chunk in acc:
                await queue.put(audio_chunk)

            await queue.put(None)  # 使用 None 标记任务的结束

        text_task = asyncio.create_task(text_generation_task())
        audio_task = asyncio.create_task(audio_generation_task())

        asyncio.gather(text_task, audio_task)

        while True:
            item = await queue.get()
            if item is None:
                break

            if isinstance(item, dict):
                await websocket.send_json(item)
            elif isinstance(item, bytes):
                await websocket.send_bytes(item)

        await websocket.send_json({"event": "done", "data": {"status": "success"}})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client.")
        if acc:
            await acc.flush()
    except Exception as e:
        logger.error(
            f"Error during WebSocket communication: {e}\n{traceback.format_exc()}"
        )
        error_payload = {"error": type(e).__name__, "detail": str(e)}
        await websocket.send_json({"event": "error", "data": error_payload})
        if acc:
            await acc.flush()
    finally:
        await websocket.close()
