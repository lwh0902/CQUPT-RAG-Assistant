"""Small MCP Streamable HTTP adapters for read-only Agent tools."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx

from services.evidence import EvidenceSource, normalize_evidence


class MCPStreamableHTTPTool:
    """Generic read-only MCP tool caller used by weather and schedule adapters."""

    def __init__(self, endpoint: str, *, tool_name: str, authorization: Optional[str] = None, timeout_seconds: float = 8.0, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self._delegate = MCPStreamableHTTPWebSearchTool(endpoint, tool_name=tool_name, authorization=authorization, timeout_seconds=timeout_seconds, transport=transport)

    async def call(self, arguments: dict[str, Any]) -> str:
        try:
            async with httpx.AsyncClient(timeout=self._delegate.timeout_seconds, transport=self._delegate.transport) as client:
                session_id, version = await self._delegate._initialize(client)
                await self._delegate._notify_initialized(client, session_id, version)
                result = await self._delegate._call_tool(client, session_id, version, arguments)
        except (httpx.HTTPError, ValueError, TypeError, KeyError):
            return json.dumps({"status": "error", "code": "TOOL_UNAVAILABLE", "message": "工具服务暂不可用"}, ensure_ascii=False)
        content = result.get("content") or []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return str(item.get("text") or "")
        return json.dumps({"status": "error", "code": "TOOL_INVALID_RESPONSE", "message": "工具未返回可用结果"}, ensure_ascii=False)


class MCPStreamableHTTPWebSearchTool:
    """Calls an MCP web-search tool and accepts only citeable results."""

    def __init__(
        self,
        endpoint: str,
        *,
        tool_name: str = "web_search",
        authorization: Optional[str] = None,
        timeout_seconds: float = 8.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.endpoint = endpoint
        self.tool_name = tool_name
        self.authorization = authorization
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def search(self, query: str, limit: int = 5) -> list[EvidenceSource]:
        normalized_query = query.strip()
        if not normalized_query or limit <= 0:
            return []

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                session_id, protocol_version = await self._initialize(client)
                await self._notify_initialized(client, session_id, protocol_version)
                result = await self._call_tool(
                    client,
                    session_id,
                    protocol_version,
                    {"query": normalized_query, "limit": min(limit, 10)},
                )
        except (httpx.HTTPError, ValueError, TypeError, KeyError):
            logging.exception("MCP web-search tool call failed")
            return []

        return _extract_web_evidence(result)

    async def _initialize(self, client: httpx.AsyncClient) -> tuple[Optional[str], str]:
        response = await client.post(
            self.endpoint,
            headers=self._headers("initialize"),
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "cqupt-rag", "version": "1.0"},
                },
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise ValueError("MCP initialize returned an error")
        version = payload.get("result", {}).get("protocolVersion", "2025-06-18")
        return response.headers.get("Mcp-Session-Id"), str(version)

    async def _notify_initialized(
        self,
        client: httpx.AsyncClient,
        session_id: Optional[str],
        protocol_version: str,
    ) -> None:
        response = await client.post(
            self.endpoint,
            headers=self._headers(
                "notifications/initialized",
                session_id=session_id,
                protocol_version=protocol_version,
            ),
            json={
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            },
        )
        response.raise_for_status()

    async def _call_tool(
        self,
        client: httpx.AsyncClient,
        session_id: Optional[str],
        protocol_version: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        response = await client.post(
            self.endpoint,
            headers=self._headers(
                "tools/call",
                session_id=session_id,
                protocol_version=protocol_version,
                tool_name=self.tool_name,
            ),
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": self.tool_name, "arguments": arguments},
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise ValueError("MCP tools/call returned an error")
        return payload.get("result", {})

    def _headers(
        self,
        method: str,
        *,
        session_id: Optional[str] = None,
        protocol_version: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "Mcp-Method": method,
        }
        if self.authorization:
            headers["Authorization"] = self.authorization
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        if protocol_version:
            headers["MCP-Protocol-Version"] = protocol_version
        if tool_name:
            headers["Mcp-Name"] = tool_name
        return headers


def _extract_web_evidence(result: dict[str, Any]) -> list[EvidenceSource]:
    content = result.get("content", [])
    if not isinstance(content, list):
        return []

    records: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        try:
            parsed = json.loads(str(item.get("text") or ""))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates = parsed.get("results", [])
        else:
            candidates = parsed
        if isinstance(candidates, list):
            records.extend(item for item in candidates if isinstance(item, dict))

    evidence: list[EvidenceSource] = []
    for record in records:
        url = str(record.get("url") or "").strip()
        host = urlsplit(url).netloc
        source = normalize_evidence(
            {
                "id": f"mcp-web:{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}",
                "source_type": "web",
                "title": record.get("title") or host,
                "snippet": record.get("snippet") or record.get("content") or "",
                "url": url,
                "site_name": record.get("site_name") or host,
                "published_at": record.get("published_at"),
                "relevance_score": record.get("relevance_score", 0.7),
                "authority_score": record.get("authority_score", 0.6),
            }
        )
        if source is not None:
            evidence.append(source)
    return evidence
