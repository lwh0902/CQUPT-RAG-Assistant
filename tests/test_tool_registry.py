import pytest

from services.evidence import EvidenceSource
from services.tool_registry import MCPFirstWebSearchTool, ToolRegistry


class FakeSearchTool:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    async def search(self, query: str, limit: int = 5):
        self.calls += 1
        return self.results


@pytest.mark.asyncio
async def test_registry_exposes_web_search_as_an_agent_tool() -> None:
    source = EvidenceSource(
        id="web-1",
        source_type="web",
        title="官网",
        snippet="通知内容",
        url="https://cqupt.edu.cn/notice",
    )
    registry = ToolRegistry()
    registry.register("web_search", FakeSearchTool([source]))

    result = await registry.search_web("奖学金")

    assert result == [source]


@pytest.mark.asyncio
async def test_mcp_first_tool_uses_native_compatibility_tool_only_when_mcp_has_no_result() -> None:
    fallback_source = EvidenceSource(
        id="web-2",
        source_type="web",
        title="官网",
        snippet="通知内容",
        url="https://cqupt.edu.cn/notice",
    )
    mcp = FakeSearchTool([])
    native = FakeSearchTool([fallback_source])

    tool = MCPFirstWebSearchTool(mcp=mcp, native_fallback=native)

    assert await tool.search("奖学金") == [fallback_source]
    assert mcp.calls == 1
    assert native.calls == 1
