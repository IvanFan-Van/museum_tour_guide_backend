"""
通过 LangGraph API (http://127.0.0.1:2024) 测试 graph 运行。

单次测试参数：
  - messages: [{"role": "user", "content": "你可以介绍一下中国瓷器吗"}]
  - language: "zh"
  - section_idx: None

运行方法：
  # 单次流式测试（默认）
  uv run tests/test_api_run.py

  # 单次同步等待测试
  uv run tests/test_api_run.py --mode wait

  # 批量测试（从 jsonl 文件读取）
  uv run tests/test_api_run.py --batch tests/batch_input.jsonl

  # 批量测试并指定输出文件
  uv run tests/test_api_run.py --batch tests/batch_input.jsonl --output results.md

JSONL 文件格式：
  - 每行一个 JSON 对象，含 query / language / section_idx 三个字段
  - 以 // 开头的行视为注释，忽略
  - 连续的 JSON 行归属同一 thread（模拟追问）
  - 超过一个空行分隔不同 thread
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

BASE_URL = "http://127.0.0.1:2024"

# ── 单次测试默认输入 ─────────────────────────────────────────
DEFAULT_PAYLOAD = {
    "messages": [{"role": "user", "content": "你可以介绍一下中国瓷器吗"}],
    "language": "zh",
    "section_idx": None,
}

# ────────────────────────────────────────────────────────────
# 底层 API 封装
# ────────────────────────────────────────────────────────────


def create_thread() -> str:
    """创建一个新的对话线程，返回 thread_id。"""
    resp = requests.post(f"{BASE_URL}/threads", json={})
    resp.raise_for_status()
    thread_id = resp.json()["thread_id"]
    print(f"  [Thread created] {thread_id}")
    return thread_id


def _make_input(query: str, language: str, section_idx: int | None) -> dict:
    """将单条 query 包装成 graph 输入 payload。"""
    return {
        "messages": [{"role": "user", "content": query}],
        "language": language,
        "section_idx": section_idx,
    }


def _extract_last_ai_message(messages: list[dict]) -> str:
    """从 messages 列表中提取最后一条 AI 消息的文本内容。"""
    for msg in reversed(messages):
        role = msg.get("type") or msg.get("role", "")
        if role in ("ai", "assistant"):
            content = msg.get("content", "")
            if isinstance(content, list):
                # tool_use / text block 列表
                texts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "\n".join(texts)
            return content
    return ""


def api_wait(thread_id: str, input_payload: dict) -> dict:
    """同步等待单次 run 完成，返回最终 state。"""
    url = f"{BASE_URL}/threads/{thread_id}/runs/wait"
    body = {"assistant_id": "graph", "input": input_payload}
    resp = requests.post(url, json=body)
    resp.raise_for_status()
    return resp.json()


def api_stream(thread_id: str, input_payload: dict) -> None:
    """以 SSE 流式方式运行，将每步的最新消息打印到控制台。"""
    url = f"{BASE_URL}/threads/{thread_id}/runs/stream"
    body = {
        "assistant_id": "graph",
        "input": input_payload,
        "stream_mode": "values",
    }
    print(f"\n  [Streaming] POST {url}")
    print(f"  [Input] {json.dumps(input_payload, ensure_ascii=False)}\n")
    print("  " + "-" * 56)

    with requests.post(url, json=body, stream=True) as resp:
        resp.raise_for_status()
        event_type: str | None = None
        for raw_line in resp.iter_lines():
            if not raw_line:
                event_type = None
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if line.startswith("event:"):
                event_type = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:") :].strip()
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    data = data_str

                print(f"  [event: {event_type}]")
                if isinstance(data, dict) and "messages" in data:
                    msgs = data["messages"]
                    if msgs:
                        last = msgs[-1]
                        role = last.get("type") or last.get("role", "?")
                        content = last.get("content", "")
                        if isinstance(content, list):
                            content = json.dumps(content, ensure_ascii=False)
                        print(f"    [{role}] {content[:500]}")
                    images = data.get("retrieved_images", [])
                    if images:
                        print(f"    [retrieved_images] {images}")
                else:
                    print(f"    {json.dumps(data, ensure_ascii=False)[:300]}")
                print()

    print("  " + "-" * 56)
    print("  [Stream finished]")


# ────────────────────────────────────────────────────────────
# 单次测试
# ────────────────────────────────────────────────────────────


def test_stream() -> None:
    """单次流式测试。"""
    print("\n" + "=" * 60)
    print("TEST: single stream run")
    print("=" * 60)
    thread_id = create_thread()
    api_stream(thread_id, DEFAULT_PAYLOAD)


def test_wait() -> None:
    """单次同步等待测试。"""
    print("\n" + "=" * 60)
    print("TEST: single wait run")
    print("=" * 60)
    thread_id = create_thread()
    result = api_wait(thread_id, DEFAULT_PAYLOAD)

    print("[Final State]")
    for msg in result.get("messages", []):
        role = msg.get("type") or msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        print(f"  [{role}] {content[:500]}")

    images = result.get("retrieved_images", [])
    if images:
        print("\n[retrieved_images]")
        for img in images:
            print(f"  {img}")


# ────────────────────────────────────────────────────────────
# JSONL 解析：将文件拆分为 thread 组
# ────────────────────────────────────────────────────────────

ThreadGroup = list[dict[str, Any]]  # 每个元素: {query, language, section_idx}


def parse_jsonl(path: Path) -> list[ThreadGroup]:
    """
    解析 JSONL 文件，返回 thread 分组列表。

    规则：
    - 以 // 开头的行（去除空白后）视为注释，跳过。
    - 空白行作为分隔符；超过一行的空白才会开启新 thread。
      （连续多个空行等同于一个分隔符）
    - 连续的 JSON 行属于同一 thread。
    """
    groups: list[ThreadGroup] = []
    current: ThreadGroup = []
    blank_streak = 0  # 连续遇到的空行数

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()

            # 注释行：跳过
            if stripped.startswith("//"):
                continue

            # 空行：记录空行连续数，超过 1 行时触发 thread 切分
            if stripped == "":
                blank_streak += 1
                if blank_streak > 1 and current:
                    groups.append(current)
                    current = []
                continue

            # 非空行：重置空行计数，解析 JSON
            blank_streak = 0
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"  [WARN] 跳过无效 JSON 行: {stripped!r} ({exc})")
                continue

            # 标准化字段
            entry: dict[str, Any] = {
                "query": str(obj.get("query", "")),
                "language": str(obj.get("language", "zh")),
                "section_idx": obj.get("section_idx"),  # None 或 int
            }
            current.append(entry)

    if current:
        groups.append(current)

    return groups


# ────────────────────────────────────────────────────────────
# 批量测试
# ────────────────────────────────────────────────────────────


def run_batch(jsonl_path: Path, output_path: Path) -> None:
    """
    读取 JSONL 文件，按 thread 分组依次执行，并将结果写入 Markdown 文件。
    """
    groups = parse_jsonl(jsonl_path)
    if not groups:
        print("[ERROR] JSONL 文件中未找到有效数据。")
        sys.exit(1)

    print(f"\n[Batch] 共解析出 {len(groups)} 个 thread 组，来自 {jsonl_path}")
    print(f"[Batch] 结果将写入 {output_path}\n")

    # 结果容器：list of thread dicts
    # thread = { thread_id, turns: [ {query, language, section_idx, answer, images} ] }
    all_results: list[dict[str, Any]] = []

    for t_idx, group in enumerate(groups, start=1):
        print(f"{'=' * 60}")
        print(f"Thread {t_idx}/{len(groups)}  ({len(group)} 条 query)")
        print(f"{'=' * 60}")

        thread_id = create_thread()
        thread_record: dict[str, Any] = {"thread_id": thread_id, "turns": []}

        for q_idx, entry in enumerate(group, start=1):
            query = entry["query"]
            language = entry["language"]
            section_idx = entry["section_idx"]

            print(
                f"\n  Turn {q_idx}/{len(group)} | language={language}"
                f" | section_idx={section_idx}"
            )
            print(f"  Q: {query}")

            input_payload = _make_input(query, language, section_idx)
            try:
                state = api_wait(thread_id, input_payload)
            except requests.HTTPError as exc:
                answer = f"[ERROR] {exc}"
                images: list[str] = []
                print(f"  A: {answer}")
            else:
                answer = _extract_last_ai_message(state.get("messages", []))
                images = state.get("retrieved_images", [])
                print(f"  A: {answer[:300]}{'...' if len(answer) > 300 else ''}")
                if images:
                    print(f"  Images: {images}")

            thread_record["turns"].append(
                {
                    "query": query,
                    "language": language,
                    "section_idx": section_idx,
                    "answer": answer,
                    "images": images,
                }
            )

        all_results.append(thread_record)

    _write_markdown(all_results, output_path, jsonl_path)
    print(f"\n[Batch] 完成！结果已保存至 {output_path}")


# ────────────────────────────────────────────────────────────
# 结果输出：Markdown
# ────────────────────────────────────────────────────────────


def _write_markdown(
    results: list[dict[str, Any]], output_path: Path, source_path: Path
) -> None:
    """将批量测试结果写成可读的 Markdown 文件。"""
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# 批量测试结果\n")
    lines.append(f"- **来源文件**：`{source_path}`")
    lines.append(f"- **运行时间**：{now}")
    lines.append(f"- **Thread 数量**：{len(results)}")
    lines.append(f"- **API**：`{BASE_URL}`")
    lines.append("")

    for t_idx, thread in enumerate(results, start=1):
        thread_id = thread["thread_id"]
        turns = thread["turns"]
        lines.append(f"---\n")
        lines.append(f"## Thread {t_idx}")
        lines.append(f"`thread_id`: `{thread_id}`  ")
        lines.append(f"共 **{len(turns)}** 轮对话\n")

        for q_idx, turn in enumerate(turns, start=1):
            lang_label = "🇨🇳 中文" if turn["language"] == "zh" else "🇬🇧 English"
            sec = (
                f"section_idx = `{turn['section_idx']}`"
                if turn["section_idx"] is not None
                else "section_idx = `None`"
            )
            lines.append(f"### Turn {q_idx}  —  {lang_label}  ·  {sec}\n")
            lines.append(f"**User**\n")
            lines.append(f"> {turn['query']}\n")
            lines.append(f"**Assistant**\n")
            # 多行回答用 blockquote
            answer_lines = turn["answer"].splitlines()
            for al in answer_lines:
                lines.append(f"> {al}")
            lines.append("")
            if turn["images"]:
                lines.append(f"**Retrieved Images**\n")
                for img in turn["images"]:
                    # 尝试渲染为图片链接（如果是文件路径或 URL 则展示）
                    lines.append(f"- `{img}`")
                lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    # 批量测试（输出到自动命名的 .md 文件）
    python tests/test_api_run.py --batch tests/batch_input.jsonl

    # 批量测试 + 指定输出路径
    python tests/test_api_run.py --batch tests/batch_input.jsonl --output my_results.md

    # 单次流式（默认）
    python tests/test_api_run.py

    # 单次同步等待
    python tests/test_api_run.py --mode wait

    # batch 格式: 使用 \n\n 分隔每个 Thread
    """
    parser = argparse.ArgumentParser(description="LangGraph API 测试脚本")
    parser.add_argument(
        "--mode",
        choices=["stream", "wait"],
        default="stream",
        help="单次测试模式：stream（流式）或 wait（同步等待）",
    )
    parser.add_argument(
        "--batch",
        metavar="JSONL_FILE",
        help="批量测试：指定包含测试用例的 JSONL 文件路径",
    )
    parser.add_argument(
        "--output",
        metavar="OUTPUT_FILE",
        help="批量测试结果输出路径（默认与 JSONL 同目录，文件名加时间戳）",
    )
    args = parser.parse_args()

    if args.batch:
        jsonl_path = Path(args.batch)
        if not jsonl_path.exists():
            print(f"[ERROR] 文件不存在：{jsonl_path}")
            sys.exit(1)
        if args.output:
            output_path = Path(args.output)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = jsonl_path.with_name(f"{jsonl_path.stem}_results_{ts}.md")
        run_batch(jsonl_path, output_path)
    else:
        if args.mode == "wait":
            test_wait()
        else:
            test_stream()
