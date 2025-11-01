import asyncio
import re
from typing import AsyncIterator, List, Optional
from asyncio import Queue
from src.utils import get_logger

logger = get_logger()


class AudioAccumulator:
    """
    一个异步迭代器，用于累积文本块，在达到阈值或遇到分隔符时，
    将其转换为音频并分段产出。
    """

    def __init__(
        self,
        tts_function,
        delimiter_threshold: int = 3,
        delimiters: List[str] | None = None,
    ):
        self.tts_function = tts_function
        self.delimiter_threshold = delimiter_threshold
        self.delimiters = delimiters or ["\n\n", "\n", "。", "！", "？", "!"]
        self._buffer = ""
        self._audio_queue: Queue[Optional[str]] = Queue()
        self._finished = False

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        """从队列中获取下一段音频，如果队列为空且已结束，则停止迭代。"""
        item = await self._audio_queue.get()
        if item is None:  # None 作为结束信号
            raise StopAsyncIteration
        return item

    def _clean_text(self, text: str) -> str:
        image_pattern = r"!\[.*?\]\(.*?\)"
        return re.sub(image_pattern, "", text)

    async def _process_segment(self, segment: str):
        """在线程中调用TTS函数并把结果放入队列。"""
        if not segment or segment.isspace():
            return

        cleaned_segment = self._clean_text(segment)
        # expected logging: Process Text Chunk: This is the true fact that... (total 74 words)
        logger.info(
            f"Process Text Chunk: {cleaned_segment[:50]}... (total {len(cleaned_segment)} words)"
        )

        try:
            # 使用 to_thread 在单独的线程中运行同步的TTS函数，避免阻塞
            base64_audio = await asyncio.to_thread(self.tts_function, cleaned_segment)
            if base64_audio:
                await self._audio_queue.put(base64_audio)
        except Exception as e:
            logger.error(f"Error during TTS conversion: {e}")

    async def add_chunk(self, chunk: str):
        """添加文本块并检查是否需要处理一个段落。"""
        self._buffer += chunk

        # 寻找最靠后的分隔符作为分割点
        delimiter_count = sum(self._buffer.count(d) for d in self.delimiters)

        # 如果超过字符阈值，则处理
        if delimiter_count >= self.delimiter_threshold:
            split_pos = -1
            for d in self.delimiters:
                pos = self._buffer.rfind(d)
                if pos != -1:
                    split_pos = max(pos + len(d), split_pos)

            if split_pos != -1:
                segment_to_process = self._buffer[:split_pos]
                self._buffer = self._buffer[split_pos:]
                await self._process_segment(segment_to_process)

    async def flush(self):
        """处理缓冲区中剩余的所有文本。"""
        if self._buffer:
            await self._process_segment(self._buffer)
            self._buffer = ""
        # 发送结束信号
        await self._audio_queue.put(None)
