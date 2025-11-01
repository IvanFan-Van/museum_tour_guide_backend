import asyncio
from unittest.mock import MagicMock
import pytest
from src.accumulator import AudioAccumulator


@pytest.mark.asyncio
async def test_segmentation_on_single_sentence():
    mock_tts_function = MagicMock(return_value=b"audio_data")
    accumulator = AudioAccumulator(
        tts_function=mock_tts_function, num_sentence_cached=3
    )
    text = "这是一段没有分隔符的长文本"

    await accumulator.add_chunk(text)

    mock_tts_function.assert_not_called()

    await accumulator.flush()
    await asyncio.sleep(0.01)

    mock_tts_function.assert_called_once_with(text)
    assert accumulator._buffer == ""
    assert not accumulator._audio_queue.empty()


@pytest.mark.asyncio
async def test_segmentation_with_multiple_chunks():
    """
    测试：通过多次调用 add_chunk 累积文本，达到阈值后进行处理。
    """
    # 1. 设置
    mock_tts_function = MagicMock(return_value=b"audio_data")
    accumulator = AudioAccumulator(
        tts_function=mock_tts_function, num_sentence_cached=3
    )

    # 2. 执行
    await accumulator.add_chunk("第一句。 ")
    # 此时不应触发TTS
    mock_tts_function.assert_not_called()
    assert accumulator._buffer == "第一句。 "

    await accumulator.add_chunk("第二句。 ")
    # 此时仍不应触发TTS
    mock_tts_function.assert_not_called()
    assert accumulator._buffer == "第一句。 第二句。 "

    await accumulator.add_chunk("第三句。 第四句。 ")
    # 此时分隔符总数为4，超过阈值3，应该触发TTS
    # 分割点在最后一个分隔符之后
    await asyncio.sleep(0.01)

    # 3. 验证
    mock_tts_function.assert_called_once_with("第一句。 第二句。 第三句。 第四句。 ")
    assert accumulator._buffer == ""


@pytest.mark.asyncio
async def test_text_cleaning():
    """
    测试：在处理文本前是否能正确移除Markdown图片链接。
    """
    # 1. 设置
    mock_tts_function = MagicMock(return_value=b"audio_data")
    accumulator = AudioAccumulator(
        tts_function=mock_tts_function, num_sentence_cached=1
    )
    text_with_image = (
        "这是文本。 ![图片描述](http://example.com/image.png)这是图片后的文本。 "
    )

    # 2. 执行
    await accumulator.add_chunk(text_with_image)
    await asyncio.sleep(0.01)

    # 3. 验证
    expected_text = "这是文本。 这是图片后的文本。 "
    mock_tts_function.assert_called_once_with(expected_text)


@pytest.mark.asyncio
async def test_flush_remaining_text():
    """
    测试：flush方法是否能处理缓冲区中剩余的文本。
    """
    # 1. 设置
    mock_tts_function = MagicMock(return_value=b"audio_data")
    accumulator = AudioAccumulator(
        tts_function=mock_tts_function, num_sentence_cached=5
    )

    # 2. 执行
    # 添加的文本不足以触发分割
    await accumulator.add_chunk("This is some text that won't trigger the threshold.")
    mock_tts_function.assert_not_called()

    # 调用 flush
    await accumulator.flush()
    await asyncio.sleep(0.01)

    # 3. 验证
    # flush 应该处理缓冲区中所有剩余的文本
    mock_tts_function.assert_called_once_with(
        "This is some text that won't trigger the threshold."
    )
    assert accumulator._buffer == ""
    # 验证队列中收到了音频数据和结束信号
    assert await accumulator._audio_queue.get() == b"audio_data"
    assert await accumulator._audio_queue.get() is None


@pytest.mark.asyncio
async def test_flush_with_no_delimiters():
    """
    测试：当文本没有分隔符时，flush是否能处理整个缓冲区。
    """
    # 1. 设置
    mock_tts_function = MagicMock(return_value=b"audio_data")
    accumulator = AudioAccumulator(
        tts_function=mock_tts_function, num_sentence_cached=3
    )
    text = "这是一段没有分隔符的长文本"

    # 2. 执行
    await accumulator.add_chunk(text)
    # 不应触发TTS
    mock_tts_function.assert_not_called()

    # 调用 flush
    await accumulator.flush()
    await asyncio.sleep(0.01)

    # 3. 验证
    mock_tts_function.assert_called_once_with(text)
    assert accumulator._buffer == ""
    assert not accumulator._audio_queue.empty()


@pytest.mark.asyncio
async def test_one_image_one_sentence():
    """
    测试：文本中只有一个图片链接和一个句子时，能否正确处理。
    """
    # 1. 设置
    mock_tts_function = MagicMock(return_value=b"audio_data")
    accumulator = AudioAccumulator(
        tts_function=mock_tts_function, num_sentence_cached=1
    )

    await accumulator.add_chunk("你可以参考以下图片:\n![图片描述](http:")
    await asyncio.sleep(0.01)
    mock_tts_function.assert_called_once_with("你可以参考以下图片:\n")
    await accumulator.add_chunk("//example.com/image.png)\n\n这是图片后的文本。 ")
    await asyncio.sleep(0.01)
    mock_tts_function.assert_called_with("\n\n这是图片后的文本。 ")
