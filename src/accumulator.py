import asyncio
import re
from typing import AsyncIterator, Callable, List, Optional
from asyncio import Queue
from src.utils import get_logger

logger = get_logger()


class AudioAccumulator:
    """
    一个异步迭代器，用于累积文本块，在达到阈值或遇到分隔符时，
    将其转换为音频并分段产出。
    """

    def __init__(
        self, tts_function: Callable[[str], bytes | None], num_sentence_cached: int = 2
    ):
        self.tts_function = tts_function
        self.num_sentence_cached = num_sentence_cached
        self.pattern = re.compile(r"(.*?)[。！？!:\.\?][\n\s]", flags=re.MULTILINE)
        self._buffer = ""
        self._audio_queue: Queue[Optional[bytes]] = Queue()
        self._finished = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self

    async def __anext__(self) -> bytes:
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
            audio_chunk = await asyncio.to_thread(self.tts_function, cleaned_segment)
            if audio_chunk:
                await self._audio_queue.put(audio_chunk)
        except Exception as e:
            logger.error(f"Error during TTS conversion: {e}")

    async def add_chunk(self, chunk: str):
        """添加文本块并检查是否需要处理一个段落。"""
        self._buffer += chunk

        # 寻找最靠后的分隔符作为分割点
        # delimiter_count = sum(self._buffer.count(d) for d in self.delimiters)
        sentences = self.pattern.findall(self._buffer)
        num_sentence = len(sentences)
        # 如果超过字符阈值，则处理
        if num_sentence >= self.num_sentence_cached:
            split_pos = -1
            pos = self._buffer.rfind(sentences[-1])
            if pos == -1:
                return

            split_pos = pos + len(sentences[-1]) + 2
            segment_to_process = self._buffer[:split_pos]
            self._buffer = self._buffer[split_pos:]
            await self._process_segment(segment_to_process)

    async def flush(self):
        """处理缓冲区中剩余的所有文本。"""
        if self._buffer:
            remaining_buffer = self._buffer
            self._buffer = ""
            await self._process_segment(remaining_buffer)

        # 发送结束信号
        await self._audio_queue.put(None)
