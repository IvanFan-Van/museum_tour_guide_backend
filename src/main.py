from pathlib import Path
import asyncio
import os
import traceback
import warnings
import json
import chromadb
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import AsyncAzureOpenAI
from pydantic import BaseModel
import pydantic
from src.graph import graph
from src.utils import get_tts, get_logger
from src.accumulator import AudioAccumulator
from src.ws_schema import StatusPayload, WSStatusMessage, WSMessage, WSTextChunkMessage
from src.realtime_agent import realtime_agent_loop
from api_exception import register_exception_handlers
from dotenv import find_dotenv, load_dotenv
from src.agent import AgentInput, LangGraphAgent

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

agent = LangGraphAgent(graph)

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

        if not query:
            raise ValueError("Query parameter is required.")

        agent_input = AgentInput(query=query, section_idx=doc_id)

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


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """
    WebSocket 端点用于实时音频对话

    前端发送格式:
    1. 二进制数据: 直接发送 PCM16 音频字节
    2. JSON 文本: {"type": "audio", "data": "base64_encoded_audio"}
    3. 控制消息: {"type": "control", "action": "interrupt|clear_buffer"}

    后端返回格式:
    - 文本块: {"type": "text_chunk", "payload": {"content": "...", "is_final": false}}
    - 音频块: {"type": "audio_chunk", "payload": {"audio": "base64", "format": "pcm16"}}
    - 状态:   {"type": "status",     "payload": {"status": "...", "detail": "..."}}
    """
    await websocket.accept()

    client = AsyncAzureOpenAI(
        azure_endpoint=os.environ["AZURE_REALTIME_ENDPOINT"],
        api_key=os.environ["AZURE_REALTIME_API_KEY"],
        api_version="2024-10-01-preview",
    )

    try:
        async for message in realtime_agent_loop(websocket, client):
            if isinstance(message, WSMessage):
                await websocket.send_json(message.model_dump())
            else:
                await websocket.send_bytes(message)
    except WebSocketDisconnect:
        logger.info("Realtime WebSocket disconnected by client.")
    except Exception as e:
        logger.error(f"Realtime WebSocket error: {e}")
        try:
            error_msg = WSStatusMessage(
                payload=StatusPayload(status="error", detail=str(e))
            )
            await websocket.send_text(error_msg.model_dump_json())
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# --- Vector DB Setup ---


class JSONData(BaseModel):
    id: str
    document: str
    metadata: dict


@app.post("/api/v1/setup")
async def setup(reset: bool = False):
    """
    Update the vector database from JSON files in the data directory.

    Args:
        reset: If True, clears the database and performs a full import.
               If False (default), performs an incremental update based on ID.
    """
    DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
    CHROMA_DB_DIR = Path(os.getenv("CHROMA_DB_DIR", "chroma_db"))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "museum_guide")

    logger.info(f"Starting setup. Reset={reset}. Data Dir={DATA_DIR}")

    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "message": f"Data directory '{DATA_DIR}' not found.",
            },
        )

    def process_metadata(meta):
        return {
            k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
            for k, v in meta.items()
        }

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

        if reset:
            try:
                client.delete_collection(COLLECTION_NAME)
                logger.info(f"Deleted collection '{COLLECTION_NAME}'")
            except ValueError:
                pass
            collection = client.create_collection(name=COLLECTION_NAME)
        else:
            collection = client.get_or_create_collection(name=COLLECTION_NAME)

        file_data_map = {}
        for file_path in DATA_DIR.rglob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    JSONData.model_validate(data)
                    file_data_map[data["id"]] = data
            except pydantic.ValidationError:
                logger.warning(
                    f"Skipping invalid JSON file: {file_path}. Missing required fields."
                )
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")

        file_ids = set(file_data_map.keys())

        if reset:
            ids_to_add = list(file_ids)
            if ids_to_add:
                collection.add(
                    ids=ids_to_add,
                    documents=[file_data_map[i]["document"] for i in ids_to_add],
                    metadatas=[
                        process_metadata(file_data_map[i]["metadata"])
                        for i in ids_to_add
                    ],  # type: ignore
                )
            return {
                "status": "success",
                "message": f"Full reset complete. Added {len(ids_to_add)} documents.",
            }

        else:
            existing_ids = set(collection.get()["ids"])
            ids_to_delete = list(existing_ids - file_ids)
            ids_to_add = list(file_ids - existing_ids)

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents.")

            if ids_to_add:
                collection.add(
                    ids=ids_to_add,
                    documents=[file_data_map[i]["document"] for i in ids_to_add],
                    metadatas=[
                        process_metadata(file_data_map[i]["metadata"])
                        for i in ids_to_add
                    ],  # type: ignore
                )
                logger.info(f"Added {len(ids_to_add)} documents.")

            return {
                "status": "success",
                "message": "Incremental update complete.",
                "added": len(ids_to_add),
                "deleted": len(ids_to_delete),
                "total": len(file_ids),
            }

    except Exception as e:
        logger.error(f"Setup failed: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )
