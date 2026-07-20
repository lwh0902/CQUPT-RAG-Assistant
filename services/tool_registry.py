"""Agent tool registry with MCP-first web-search selection."""

from __future__ import annotations

from typing import Optional, Protocol

from services.evidence import EvidenceSource


class WebSearchTool(Protocol):
    async def search(self, query: str, limit: int = 5) -> list[EvidenceSource]:
        """Return web evidence from a real search tool."""


class MCPFirstWebSearchTool:
    """Prefer an MCP tool and retain a native adapter for compatibility."""

    def __init__(
        self,
        *,
        mcp: Optional[WebSearchTool],
        native_fallback: Optional[WebSearchTool],
    ) -> None:
        self.mcp = mcp
        self.native_fallback = native_fallback

    async def search(self, query: str, limit: int = 5) -> list[EvidenceSource]:
        if self.mcp is not None:
            results = await self.mcp.search(query, limit)
            if results:
                return results
        if self.native_fallback is not None:
            return await self.native_fallback.search(query, limit)
        return []


class ToolRegistry:
    """Small registry that keeps orchestration independent of tool transports."""

    def __init__(self) -> None:
        self._web_search: Optional[WebSearchTool] = None

    def register(self, name: str, tool: WebSearchTool) -> None:
        if name != "web_search":
            raise ValueError(f"Unsupported evidence tool: {name}")
        self._web_search = tool

    async def search_web(self, query: str, limit: int = 5) -> list[EvidenceSource]:
        if self._web_search is None:
            return []
        return await self._web_search.search(query, limit)
