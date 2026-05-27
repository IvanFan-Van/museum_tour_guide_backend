import asyncio
import os
import traceback
import warnings
from typing import cast

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.accumulator import AudioAccumulator
from src.agent import AgentInput, LangGraphRemoteAgent
from src.models import (
    StatusPayload,
    WSMessage,
    WSStatusMessage,
    WSTextChunkMessage,
)
from src.utils import get_logger, get_tts

load_dotenv(find_dotenv())

# os.environ["ANONYMIZED_TELEMETRY"] = "False"
# warnings.filterwarnings("ignore")

logger = get_logger()
origins = ["*", "http://localhost:5174"]

app = FastAPI(
    title="Museum Tour Guide API",
    description="API for the Museum Tour Guide application using RAG with LangGraph and LangChain",
    version="1.0.0",
)

_langgraph_url = os.environ.get("LANGGRAPH_URL", "http://localhost:8123")
agent = LangGraphRemoteAgent(url=_langgraph_url, graph_name="graph")

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


async def _run_agent(
    agent_input: AgentInput, queue: asyncio.Queue, acc: AudioAccumulator
):
    print(f"Running agent with input: {agent_input}")
    async for message in agent.stream(agent_input):
        await queue.put(message.model_dump())  # type: ignore
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
            websocket.send_json(
                WSStatusMessage(
                    payload=StatusPayload(
                        status="error", detail="Query parameter is required."
                    )
                ).model_dump()  # type: ignore
            )
            return await websocket.close()

        if not thread_id:
            websocket.send_json(
                WSStatusMessage(
                    payload=StatusPayload(
                        status="error", detail="Session ID (thread_id) is required."
                    )
                ).model_dump()  # type: ignore
            )
            return await websocket.close()

        agent_input = AgentInput(
            query=query, thread_id=thread_id, section_idx=doc_id, language=language
        )
        queue: asyncio.Queue[dict | bytes] = asyncio.Queue()
        acc = AudioAccumulator(tts_function=get_tts(), num_sentence_cached=1)

        # 2. 并行执行图计算和 TTS 生成
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_run_agent(agent_input, queue, acc))
            tg.create_task(_run_tts(acc, queue))

        # 3. 返回结果
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, dict):
                await websocket.send_json(item)
            elif isinstance(item, bytes):
                await websocket.send_bytes(item)

    except ExceptionGroup as eg:
        for e in eg.exceptions:
            logger.error(f"Error in concurrent tasks: {e}\n{traceback.format_exc()}")
        await websocket.send_json(
            WSStatusMessage(
                payload=StatusPayload(
                    status="error",
                    detail="An error occurred during processing. Please try again.",
                )
            ).model_dump()  # type: ignore
        )

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
