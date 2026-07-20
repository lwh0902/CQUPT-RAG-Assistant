import pytest

from services.evidence import EvidenceSource
from services.retrieval import collect_evidence, decide_local_evidence


class FakeToolRegistry:
    def __init__(self, web_results):
        self.web_results = web_results
        self.calls = 0

    async def search_web(self, query: str, limit: int = 5):
        self.calls += 1
        return self.web_results


def test_decide_local_evidence_rejects_low_vector_and_absent_bm25_scores() -> None:
    assert decide_local_evidence(
        has_local_source=True,
        vector_score=0.12,
        bm25_score=5.71,
        keyword_coverage=0.0,
    ) == "out_of_scope"


@pytest.mark.asyncio
async def test_collect_evidence_does_not_call_web_tool_when_switch_is_off(monkeypatch) -> None:
    async def fake_rag_context(query, retriever):
        return "【学生手册｜第 12 页】\n国家奖学金奖励标准为 10000 元。"

    monkeypatch.setattr("services.retrieval.get_rag_context_async", fake_rag_context)
    tools = FakeToolRegistry([])

    result = await collect_evidence("奖学金", retriever=object(), tools=tools, web_search_enabled=False)

    assert tools.calls == 0
    assert [source.source_type for source in result.sources] == ["knowledge_base"]


@pytest.mark.asyncio
async def test_collect_evidence_merges_local_and_mcp_tool_results_when_enabled(monkeypatch) -> None:
    async def fake_rag_context(query, retriever):
        return "【学生手册｜第 12 页】\n国家奖学金奖励标准为 10000 元。"

    monkeypatch.setattr("services.retrieval.get_rag_context_async", fake_rag_context)
    web = EvidenceSource(
        id="web-1",
        source_type="web",
        title="学校官网通知",
        snippet="国家奖学金奖励标准为 10000 元。",
        url="https://cqupt.edu.cn/notice/1",
        site_name="cqupt.edu.cn",
        relevance_score=0.9,
        authority_score=0.95,
    )
    tools = FakeToolRegistry([web])

    result = await collect_evidence("奖学金", retriever=object(), tools=tools, web_search_enabled=True)

    assert tools.calls == 1
    assert {source.source_type for source in result.sources} == {"knowledge_base", "web"}


@pytest.mark.asyncio
async def test_collect_evidence_marks_missing_local_knowledge_as_out_of_scope(monkeypatch) -> None:
    async def fake_rag_context(query, retriever):
        return ""

    monkeypatch.setattr("services.retrieval.get_rag_context_async", fake_rag_context)
    result = await collect_evidence("图书馆今天有座位吗", retriever=object(), tools=FakeToolRegistry([]), web_search_enabled=False)

    assert result.decision == "out_of_scope"
    assert result.sources == []


@pytest.mark.asyncio
async def test_collect_evidence_hides_weak_local_sources_for_an_out_of_scope_query(monkeypatch) -> None:
    async def fake_rag_context(query, retriever):
        return "【学生手册｜第 12 页】\n国家奖学金奖励标准为 10000 元。"

    async def fake_probe(query, retriever, context):
        return context, 0.02, 5.71, 0.0

    monkeypatch.setattr("services.retrieval.get_rag_context_async", fake_rag_context)
    monkeypatch.setattr("services.retrieval._probe_local_retrieval", fake_probe)

    result = await collect_evidence("图书馆今天有座位吗", retriever=object(), tools=None, web_search_enabled=False)

    assert result.decision == "out_of_scope"
    assert result.sources == []
