import json

import httpx
import pytest

from services.mcp_tools import MCPStreamableHTTPTool


@pytest.mark.asyncio
async def test_generic_mcp_tool_returns_structured_tool_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "initialize":
            return httpx.Response(200, headers={"Mcp-Session-Id": "s1"}, json={"jsonrpc": "2.0", "id": 1, "result": {}})
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": '{"status":"success"}'}]}})

    tool = MCPStreamableHTTPTool("https://mcp.example/mcp", tool_name="weather.query", transport=httpx.MockTransport(handler))
    assert await tool.call({"city_name": "重庆"}) == '{"status":"success"}'
