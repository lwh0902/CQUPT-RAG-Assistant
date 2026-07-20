"""Real web-search providers used only when a user enables web search."""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional, Protocol
from urllib.parse import urlsplit

import httpx

from services.evidence import EvidenceSource, normalize_evidence
from services.mcp_tools import MCPStreamableHTTPWebSearchTool
from services.tool_registry import MCPFirstWebSearchTool, ToolRegistry, WebSearchTool


class WebSearchProvider(Protocol):
    async def search(self, query: str, limit: int = 5) -> list[EvidenceSource]:
        """Return verifiable web evidence for a query."""


class DisabledWebSearchProvider:
    """Explicit no-op provider used when no real search credential is configured."""

    async def search(self, query: str, limit: int = 5) -> list[EvidenceSource]:
        return []


class TavilyWebSearchProvider:
    """Minimal Tavily adapter that never fabricates search results."""

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = "https://api.tavily.com/search",
        timeout_seconds: float = 8.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def search(self, query: str, limit: int = 5) -> list[EvidenceSource]:
        normalized_query = query.strip()
        if not normalized_query or limit <= 0:
            return []

        payload = {
            "api_key": self.api_key,
            "query": normalized_query,
            "max_results": min(limit, 10),
            "search_depth": "basic",
            "include_answer": False,
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(self.endpoint, json=payload)
                response.raise_for_status()
                results = response.json().get("results", [])
        except (httpx.HTTPError, ValueError, TypeError):
            logging.exception("Web search request failed")
            return []

        evidence: list[EvidenceSource] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            url = str(result.get("url") or "").strip()
            host = urlsplit(url).netloc
            source = normalize_evidence(
                {
                    "id": f"web:{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}",
                    "source_type": "web",
                    "title": result.get("title") or host,
                    "snippet": result.get("content") or result.get("snippet") or "",
                    "url": url,
                    "site_name": host,
                    "published_at": result.get("published_date"),
                    "relevance_score": result.get("score", 0.6),
                    "authority_score": _authority_score(host),
                }
            )
            if source is not None:
                evidence.append(source)

        return evidence


def get_web_search_provider() -> WebSearchProvider:
    provider = os.getenv("WEB_SEARCH_PROVIDER", "disabled").strip().lower()
    if provider != "tavily":
        return DisabledWebSearchProvider()

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        logging.warning("WEB_SEARCH_PROVIDER=tavily but TAVILY_API_KEY is missing")
        return DisabledWebSearchProvider()

    return TavilyWebSearchProvider(
        api_key,
        timeout_seconds=float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "8")),
    )


def build_tool_registry() -> ToolRegistry:
    """Build the web-search tool once during application startup."""
    registry = ToolRegistry()
    mcp_endpoint = os.getenv("MCP_WEB_SEARCH_URL", "").strip()
    mcp_tool: Optional[WebSearchTool] = None
    if mcp_endpoint:
        token = os.getenv("MCP_WEB_SEARCH_AUTH_TOKEN", "").strip()
        authorization = f"Bearer {token}" if token else None
        mcp_tool = MCPStreamableHTTPWebSearchTool(
            mcp_endpoint,
            tool_name=os.getenv("MCP_WEB_SEARCH_TOOL_NAME", "web_search").strip() or "web_search",
            authorization=authorization,
            timeout_seconds=float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "8")),
        )

    registry.register(
        "web_search",
        MCPFirstWebSearchTool(
            mcp=mcp_tool,
            native_fallback=get_web_search_provider(),
        ),
    )
    return registry


def _authority_score(host: str) -> float:
    normalized = host.lower()
    if normalized.endswith(".cqupt.edu.cn") or normalized == "cqupt.edu.cn":
        return 0.95
    if normalized.endswith(".gov.cn") or normalized.endswith(".edu.cn"):
        return 0.85
    return 0.6
