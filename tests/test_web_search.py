import httpx
import pytest

from services.web_search import DisabledWebSearchProvider, TavilyWebSearchProvider


@pytest.mark.asyncio
async def test_disabled_provider_never_returns_model_or_fake_web_results() -> None:
    provider = DisabledWebSearchProvider()

    assert await provider.search("重庆邮电大学奖学金") == []


@pytest.mark.asyncio
async def test_tavily_provider_only_maps_results_with_real_urls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://api.tavily.test/search")
        assert request.json() if hasattr(request, "json") else request.content
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://cqupt.edu.cn/notice/1",
                        "title": "奖学金通知",
                        "content": "校内正式通知正文摘要",
                        "published_date": "2026-07-01",
                    },
                    {"url": "", "title": "无链接结果", "content": "不能作为来源"},
                ]
            },
        )

    provider = TavilyWebSearchProvider(
        api_key="test-key",
        endpoint="https://api.tavily.test/search",
        transport=httpx.MockTransport(handler),
    )

    results = await provider.search("奖学金", limit=3)

    assert len(results) == 1
    assert results[0].source_type == "web"
    assert results[0].url == "https://cqupt.edu.cn/notice/1"
    assert results[0].published_at == "2026-07-01"


@pytest.mark.asyncio
async def test_tavily_timeout_degrades_to_empty_evidence() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("search timeout")

    provider = TavilyWebSearchProvider(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )

    assert await provider.search("奖学金") == []
