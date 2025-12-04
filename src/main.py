from pathlib import Path
import asyncio
import os
import traceback
from typing import TypedDict
import warnings
import json
import chromadb
from langchain_core.runnables.schema import StreamEvent
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pydantic
from src.graph import graph
from src.utils import get_tts, get_logger
from src.accumulator import AudioAccumulator
from api_exception import register_exception_handlers
from dotenv import find_dotenv, load_dotenv

# import .env
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
                else:
                    logger.debug(f"Ignored event: {event}")
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


class JSONData(BaseModel):
    id: str
    document: str
    metadata: dict


@app.post("/api/v1/setup")
async def setup(reset: bool = False):
    """
    Update the vector database from JSON files in the data directory.

    Args:
        reset (bool): If True, clears the database and performs a full import.
                      If False (default), performs an incremental update based on ID.
    """
    # Configuration - using environment variables with defaults
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

    try:
        # Initialize Chroma Client
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

        # Handle Reset
        if reset:
            try:
                client.delete_collection(COLLECTION_NAME)
                logger.info(f"Deleted collection {COLLECTION_NAME}")
            except ValueError:
                pass  # Collection might not exist
            collection = client.create_collection(name=COLLECTION_NAME)
        else:
            collection = client.get_or_create_collection(name=COLLECTION_NAME)

        # Read files
        json_files = list(DATA_DIR.rglob("*.json"))

        file_data_map = {}  # Map id -> data

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Validate required fields
                    JSONData.model_validate(data)
                    file_data_map[data["id"]] = data
            except pydantic.ValidationError:
                logger.warning(
                    f"Skipping invalid JSON file: {file_path}. Missing required fields."
                )
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")

        file_ids = set(file_data_map.keys())

        # Helper to process metadata (convert lists to strings for Chroma)
        def process_metadata(meta):
            new_meta = {}
            for k, v in meta.items():
                if isinstance(v, (list, dict)):
                    new_meta[k] = json.dumps(v, ensure_ascii=False)
                else:
                    new_meta[k] = v
            return new_meta

        if reset:
            # Add all
            ids_to_add = list(file_ids)
            if ids_to_add:
                documents = [file_data_map[i]["document"] for i in ids_to_add]
                metadatas = [
                    process_metadata(file_data_map[i]["metadata"]) for i in ids_to_add
                ]
                collection.add(ids=ids_to_add, documents=documents, metadatas=metadatas)  # type: ignore

            return {
                "status": "success",
                "message": f"Full reset complete. Added {len(ids_to_add)} documents.",
            }

        else:
            # Incremental
            existing_ids = set(collection.get()["ids"])

            ids_to_delete = list(existing_ids - file_ids)
            ids_to_add = list(file_ids - existing_ids)

            # Delete
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents.")

            # Add
            if ids_to_add:
                documents = [file_data_map[i]["document"] for i in ids_to_add]
                metadatas = [
                    process_metadata(file_data_map[i]["metadata"]) for i in ids_to_add
                ]

                collection.add(ids=ids_to_add, documents=documents, metadatas=metadatas)  # type: ignore
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
