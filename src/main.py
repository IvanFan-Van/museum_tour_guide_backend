import asyncio
import os
import traceback
from typing import cast
import warnings
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.utils import get_tts, get_logger
from src.accumulator import AudioAccumulator
from src.models import (
    WSTextChunkMessage,
)
from api_exception import register_exception_handlers
from dotenv import find_dotenv, load_dotenv
from src.agent import AgentInput, LangGraphRemoteAgent

load_dotenv(find_dotenv())

os.environ["ANONYMIZED_TELEMETRY"] = "False"

warnings.filterwarnings("ignore")
logger = get_logger()
origins = ["*", "http://localhost:5174"]

app = FastAPI(
    title="Museum Tour Guide API",
    description="API for the Museum Tour Guide application using RAG with LangGraph and LangChain",
    version="1.0.0",
)

_langgraph_url = os.environ.get("LANGGRAPH_URL", "http://localhost:8123")
agent = LangGraphRemoteAgent(url=_langgraph_url, graph_name="graph")

register_exception_handlers(app, log_traceback=False, log=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return {"message": "Welcome to the Museum Tour Guide API!!!"}


@app.get("/api/v1/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/v1/error")
async def error():
    raise ValueError("This is a test error endpoint.")


async def _run_agent(
    agent_input: AgentInput, queue: asyncio.Queue, acc: AudioAccumulator
):
    async for message in agent.stream(agent_input):
        message = cast(BaseModel, message)
        await queue.put(message.model_dump())
        if isinstance(message, WSTextChunkMessage):
            if message.payload.content and not message.payload.is_final:
                await acc.add_chunk(message.payload.content)
    await acc.flush()


async def _run_tts(acc: AudioAccumulator, queue: asyncio.Queue):
    async for audio_chunk in acc:
        await queue.put(audio_chunk)
    await queue.put(None)  # 结束信号


@app.websocket("/api/v1/invoke")
async def invoke(websocket: WebSocket):
    await websocket.accept()
    acc = None

    try:
        # 1. 数据校验
        data = await websocket.receive_json()
        query = data.get("query", None)
        doc_id = data.get("doc_id", None)
        thread_id = data.get("session_id", None)
        language = data.get("language", "en")

        if not query:
            raise ValueError("Query parameter is required.")
        if not thread_id:
            raise ValueError("thread_id parameter is required.")

        agent_input = AgentInput(
            query=query, thread_id=thread_id, section_idx=doc_id, language=language
        )

        queue = asyncio.Queue()
        acc = AudioAccumulator(tts_function=get_tts(), num_sentence_cached=1)

        # 2. 并行执行图计算和 TTS 生成
        graph_task = asyncio.create_task(_run_agent(agent_input, queue, acc))
        audio_task = asyncio.create_task(_run_tts(acc, queue))
        asyncio.gather(graph_task, audio_task)

        # 3. 返回结果
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, dict):
                await websocket.send_json(item)
            elif isinstance(item, bytes):
                await websocket.send_bytes(item)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client.")
        if acc:
            await acc.flush()
    except Exception as e:
        logger.error(
            f"Error during WebSocket communication: {e}\n{traceback.format_exc()}"
        )
        await websocket.send_json(
            {"event": "error", "data": {"error": type(e).__name__, "detail": str(e)}}
        )
        if acc:
            await acc.flush()
    finally:
        await websocket.close()


# @app.websocket("/ws/realtime")
# async def websocket_realtime(websocket: WebSocket):
#     """
#     WebSocket 端点用于实时音频对话

#     前端发送格式:
#     1. 二进制数据: 直接发送 PCM16 音频字节
#     2. JSON 文本: {"type": "audio", "data": "base64_encoded_audio"}
#     3. 控制消息: {"type": "control", "action": "interrupt|clear_buffer"}

#     后端返回格式:
#     - 文本块: {"type": "text_chunk", "payload": {"content": "...", "is_final": false}}
#     - 音频块: {"type": "audio_chunk", "payload": {"audio": "base64", "format": "pcm16"}}
#     - 状态:   {"type": "status",     "payload": {"status": "...", "detail": "..."}}
#     """
#     await websocket.accept()

#     client = AsyncAzureOpenAI(
#         azure_endpoint=os.environ["AZURE_REALTIME_ENDPOINT"],
#         api_key=os.environ["AZURE_REALTIME_API_KEY"],
#         api_version="2024-10-01-preview",
#     )

#     try:
#         async for message in realtime_agent_loop(websocket, client):
#             if isinstance(message, WSMessage):
#                 await websocket.send_json(message.model_dump())
#             else:
#                 await websocket.send_bytes(message)
#     except WebSocketDisconnect:
#         logger.info("Realtime WebSocket disconnected by client.")
#     except Exception as e:
#         logger.error(f"Realtime WebSocket error: {e}")
#         try:
#             error_msg = WSStatusMessage(
#                 payload=StatusPayload(status="error", detail=str(e))
#             )
#             await websocket.send_text(error_msg.model_dump_json())
#         except Exception:
#             pass
#     finally:
#         try:
#             await websocket.close()
#         except Exception:
#             pass
