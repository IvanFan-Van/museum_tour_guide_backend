"""Realtime 语音对话 agent loop"""

import asyncio
import base64
import json
import os
from typing import AsyncGenerator, Union

from fastapi import WebSocket, WebSocketDisconnect
from openai import AsyncAzureOpenAI

from src.ws_schema import (
    StatusPayload,
    TextChunkPayload,
    WSStatusMessage,
    WSTextChunkMessage,
    WSMessage,
)


async def realtime_agent_loop(
    websocket: WebSocket,
    client: AsyncAzureOpenAI,
) -> AsyncGenerator[Union[WSTextChunkMessage, WSStatusMessage, bytes], None]:
    """处理实时音频对话的主循环"""
    try:
        async with client.beta.realtime.connect(model="gpt-realtime") as rt:
            await rt.session.update(
                session={
                    "modalities": ["text", "audio"],
                    "instructions": "You are a knowledgeable museum tour guide. Provide engaging and informative explanations about artworks and exhibits in a conversational manner.",
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.6,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 800,
                    },
                }
            )

            async def receive_audio_from_frontend():
                """接收前端发送的音频数据"""
                try:
                    while True:
                        data = await websocket.receive()
                        if "bytes" in data:
                            audio_base64 = base64.b64encode(data["bytes"]).decode(
                                "utf-8"
                            )
                            await rt.input_audio_buffer.append(audio=audio_base64)
                        elif "text" in data:
                            message = json.loads(data["text"])
                            if message.get("type") == "audio":
                                await rt.input_audio_buffer.append(
                                    audio=message["data"]
                                )
                            elif message.get("type") == "control":
                                action = message.get("action")
                                if action == "interrupt":
                                    await rt.response.cancel()
                                elif action == "clear_buffer":
                                    await rt.input_audio_buffer.clear()
                except WebSocketDisconnect:
                    pass
                except Exception as e:
                    print(f"接收音频时出错: {e}")

            async def send_responses_to_frontend():
                """处理 Realtime API 的事件并发送到前端"""
                try:
                    async for event in rt:
                        if event.type == "response.audio_transcript.delta":
                            yield WSTextChunkMessage(
                                payload=TextChunkPayload(
                                    content=event.delta, is_final=False
                                )
                            )
                        elif event.type == "response.audio_transcript.done":
                            yield WSTextChunkMessage(
                                payload=TextChunkPayload(content="", is_final=True)
                            )
                        elif event.type == "response.audio.delta":
                            if event.delta:
                                yield base64.b64decode(event.delta)
                        elif event.type == "input_audio_buffer.speech_started":
                            yield WSStatusMessage(
                                payload=StatusPayload(
                                    status="speech_detected",
                                    detail="User speech detected, interrupting response",
                                )
                            )
                        elif event.type == "response.done":
                            yield WSStatusMessage(
                                payload=StatusPayload(
                                    status="response_completed",
                                    detail="Response generation completed",
                                )
                            )
                        elif event.type == "error":
                            yield WSStatusMessage(
                                payload=StatusPayload(
                                    status="error",
                                    detail=str(event.error),
                                )
                            )
                except Exception as e:
                    print(f"发送响应时出错: {e}")
                    yield WSStatusMessage(
                        payload=StatusPayload(status="error", detail=str(e))
                    )

            receive_task = asyncio.create_task(receive_audio_from_frontend())
            async for message in send_responses_to_frontend():
                yield message
            receive_task.cancel()

    except Exception as e:
        print(f"Realtime agent loop 错误: {e}")
        yield WSStatusMessage(payload=StatusPayload(status="error", detail=str(e)))
