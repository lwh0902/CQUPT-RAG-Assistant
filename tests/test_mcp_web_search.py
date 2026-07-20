import json

import httpx
import pytest

from services.mcp_tools import MCPStreamableHTTPWebSearchTool


@pytest.mark.asyncio
async def test_mcp_tool_initializes_then_returns_verifiable_web_evidence() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        if body["method"] == "initialize":
            assert request.headers["Mcp-Method"] == "initialize"
            return httpx.Response(
                200,
                headers={"Mcp-Session-Id": "session-1"},
                json={"jsonrpc": "2.0", "id": body["id"], "result": {"protocolVersion": "2025-06-18"}},
            )

        if body["method"] == "notifications/initialized":
            assert request.headers["Mcp-Session-Id"] == "session-1"
            return httpx.Response(202)

        assert body["method"] == "tools/call"
        assert request.headers["Mcp-Session-Id"] == "session-1"
        assert request.headers["Mcp-Method"] == "tools/call"
        assert request.headers["Mcp-Name"] == "web_search"
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "results": [
                                        {
                                            "title": "学校通知",
                                            "url": "https://cqupt.edu.cn/notice/1",
                                            "snippet": "正式通知内容",
                                            "published_at": "2026-07-15",
                                        }
                                    ]
                                }
                            ),
                        }
                    ]
                },
            },
        )

    tool = MCPStreamableHTTPWebSearchTool(
        endpoint="https://mcp.example.test/mcp",
        transport=httpx.MockTransport(handler),
    )

    evidence = await tool.search("奖学金", limit=3)

    assert len(requests) == 3
    assert evidence[0].source_type == "web"
    assert evidence[0].url == "https://cqupt.edu.cn/notice/1"


@pytest.mark.asyncio
async def test_mcp_tool_rejects_text_without_real_source_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "initialize":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {}})
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {"content": [{"type": "text", "text": "模型整理的一段话"}]},
            },
        )

    tool = MCPStreamableHTTPWebSearchTool(
        endpoint="https://mcp.example.test/mcp",
        transport=httpx.MockTransport(handler),
    )

    assert await tool.search("奖学金") == []
