import asyncio
import os
import traceback
import warnings
import json
from fastapi import FastAPI, Request
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


@app.get("/api/v1/invoke")
async def invoke(query: str, doc_id: str | None = None):
    """
    与 LangGraph RAG 应用进行交互，通过 Server-Sent Events (SSE) 流式返回结果。

    SSE 事件流包含以下几种事件类型:

    - **event: connected**
      - 描述: 连接成功后发送的第一条消息。
      - 数据结构: `{"status": "success"}`

    - **event: message**
      - 描述: RAG 应用生成的内容块，会持续发送直到内容生成完毕。
      - 数据结构: `{"type": "text" 或 "audio", "chunk": "..."}` 或 `{"type": "audio", "chunk": "data:audio/wav;base64,..."}`
        - `chunk` (str): 模型生成的文本片段或是音频片段。

    - **event: done**
      - 描述: 表示所有内容已成功生成并发送完毕。
      - 数据结构: `{"finish": true}`

    - **event: error**
      - 描述: 在处理过程中发生错误时发送。
      - 数据结构: `{"error": "...", "detail": "..."}`
        - `error` (str): 错误类型或简短描述。
        - `detail` (str): 详细的错误信息。
    """
    graph_input = {
        "messages": [{"role": "user", "content": query}],
        "doc_id": doc_id,
    }

    # 获取节点ID的辅助函数
    def get_node_id(node_name: str, event):
        if event["name"] == node_name:
            return event["run_id"]
        else:
            return None

    # 结果队列, 存储任务完成后的结果
    queue = asyncio.Queue()

    # 生成SSE响应的异步生成器
    async def generate():
        generator_id = None
        tts = get_tts()
        acc = AudioAccumulator(tts_function=tts, delimiter_threshold=3)
        try:
            # 发送 connected 事件
            yield f"event: connected\ndata: {json.dumps({'status': 'success'})}\n\n"

            # 文本生成函数
            async def text_generation_task():
                async for event in graph.astream_events(graph_input, version="v2"):
                    nonlocal generator_id
                    if generator_id is None:
                        generator_id = get_node_id("generator", event)

                    if (
                        event["event"] == "on_chat_model_stream"
                        and generator_id in event["parent_ids"]
                    ):
                        chunk = event["data"]["chunk"].content
                        if chunk:
                            data = {
                                "type": "text",
                                "chunk": chunk,
                            }
                            # 将结果添加到结果队列以及 accumulator 中
                            await queue.put(
                                f"event: message\ndata: {json.dumps(data)}\n\n"
                            )
                            await acc.add_chunk(chunk)
                await acc.flush()

            # 音频生成任务
            async def audio_generation_task():
                async for audio_chunk in acc:
                    data = {
                        "type": "audio",
                        "chunk": f"data:audio/wav;base64,{audio_chunk}",
                    }
                    await queue.put(f"event: message\ndata: {json.dumps(data)}\n\n")

                await queue.put(None)  # 使用 None 标记任务的结束

            text_task = asyncio.create_task(text_generation_task())
            audio_task = asyncio.create_task(audio_generation_task())

            asyncio.gather(text_task, audio_task)

            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

            # 发送 done 事件
            yield f"event: done\ndata: {json.dumps({'finish': True})}\n\n"
        except Exception as e:
            logger.error(f"Error during SSE generation: {e}\n{traceback.format_exc()}")
            error_payload = {"error": type(e).__name__, "detail": str(e)}
            # 发送 error 事件
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            await acc.flush()

    return StreamingResponse(generate(), media_type="text/event-stream")
