"""Unified DeepSeek LLM client and streaming helpers."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Optional

from openai import OpenAI

from settings import MODEL_NAME
from tools import AGENT_TOOLS_SCHEMA

_client: Optional[OpenAI] = None


def get_llm_client() -> OpenAI:
    """Return a singleton OpenAI-compatible DeepSeek client."""
    global _client
    if _client is None:
        import os

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("未找到 DEEPSEEK_API_KEY，请检查 .env 配置。")
        _client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    return _client


def create_llm_completion(
    messages: list[dict[str, Any]],
    *,
    with_tools: bool,
    stream: bool = False,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Any:
    request_payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": stream,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    if with_tools:
        request_payload["tools"] = AGENT_TOOLS_SCHEMA
        request_payload["tool_choice"] = "auto"
    if temperature is not None:
        request_payload["temperature"] = temperature
    if top_p is not None:
        request_payload["top_p"] = top_p
    return get_llm_client().chat.completions.create(**request_payload)


async def stream_llm_text(
    messages: list[dict[str, Any]],
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
):
    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def put_event(event_type: str, payload: str = "") -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (event_type, payload))

    def worker() -> None:
        try:
            stream_response = create_llm_completion(
                messages,
                with_tools=False,
                stream=True,
                temperature=temperature,
                top_p=top_p,
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
