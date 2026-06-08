"""Unified ZhipuAI LLM client and streaming helpers."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Optional

from zhipuai import ZhipuAI

from tools import AGENT_TOOLS_SCHEMA

_client: Optional[ZhipuAI] = None


def get_glm_client() -> ZhipuAI:
    global _client
    if _client is None:
        import os
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("未找到 ZHIPU_API_KEY，请检查 .env 配置。")
        _client = ZhipuAI(api_key=api_key)
    return _client


def create_glm_completion(
    messages: list[dict[str, Any]],
    *,
    with_tools: bool,
    stream: bool = False,
) -> Any:
    request_payload: dict[str, Any] = {
        "model": "glm-4.7-flash",
        "messages": messages,
        "stream": stream,
        "thinking": {"type": "disabled"},
    }
    if with_tools:
        request_payload["tools"] = AGENT_TOOLS_SCHEMA
        request_payload["tool_choice"] = "auto"
    return get_glm_client().chat.completions.create(**request_payload)


async def stream_glm_text(messages: list[dict[str, Any]]):
    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def put_event(event_type: str, payload: str = "") -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (event_type, payload))

    def worker() -> None:
        try:
            stream_response = create_glm_completion(
                messages,
                with_tools=False,
                stream=True,
            )
            for chunk in stream_response:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue

                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    put_event("delta", content)
        except Exception as exc:
            logging.exception("Streaming completion failed")
            put_event("error", str(exc))
        finally:
            put_event("done")

    threading.Thread(target=worker, daemon=True).start()

    while True:
        event_type, payload = await queue.get()
        if event_type == "delta":
            yield payload
            continue
        if event_type == "error":
            raise RuntimeError(payload)
        break
