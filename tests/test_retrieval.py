import pytest

from services.evidence import EvidenceSource
from services.retrieval import (
    collect_evidence,
    decide_from_docs,
    decide_local_evidence,
    is_personal_data_query,
)


class FakeToolRegistry:
    def __init__(self, web_results):
        self.web_results = web_results
        self.calls = 0

    async def search_web(self, query: str, limit: int = 5):
        self.calls += 1
        return self.web_results


def test_decide_local_evidence_rejects_low_vector_without_lexical_support() -> None:
    assert decide_local_evidence(
        has_local_source=True,
        vector_score=0.12,
        bm25_score=1.0,
        keyword_coverage=0.0,
    ) == "out_of_scope"


def test_decide_local_evidence_supports_mid_vector_with_keyword_overlap() -> None:
    assert decide_local_evidence(
        has_local_source=True,
        vector_score=0.37,
        bm25_score=6.5,
        keyword_coverage=0.5,
    ) == "supported"


def test_decide_local_evidence_supports_strong_bm25_with_mid_vector() -> None:
    assert decide_local_evidence(
        has_local_source=True,
        vector_score=0.37,
        bm25_score=6.5,
        keyword_coverage=0.0,
    ) == "supported"


def test_bm25_coverage_path_blocked_when_vector_below_floor() -> None:
    """Campus-flavored but unanswerable (library hours): BM25+coverage must not
    override a clearly weak vector match (calibrated 2026-07-22, R086-class)."""
    assert decide_local_evidence(
        has_local_source=True,
        vector_score=0.11,
        bm25_score=7.1,
        keyword_coverage=0.5,
    ) == "insufficient"


def test_bm25_coverage_path_allowed_above_floor() -> None:
    """Colloquial statute QA just above the floor stays supported (C003-class)."""
    assert decide_local_evidence(
        has_local_source=True,
        vector_score=0.29,
        bm25_score=8.9,
        keyword_coverage=0.67,
    ) == "supported"


def test_is_personal_data_query_detects_private_record_asks() -> None:
    assert is_personal_data_query("我的个人综测成绩是多少？") is True
    assert is_personal_data_query("查一下我的绩点") is True
    assert is_personal_data_query("综测成绩怎么算？") is False


def test_decide_from_docs_rejects_personal_data_even_if_docs_exist() -> None:
    fake_docs = [object()]
    assert decide_from_docs("我的个人综测成绩是多少？", retriever=object(), docs=fake_docs) == "out_of_scope"


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
async def test_collect_evidence_skips_local_kb_for_personal_data_query(monkeypatch) -> None:
    called = {"rag": 0}

    async def fake_rag_context(query, retriever):
        called["rag"] += 1
        return "【综测办法｜第 3 页】\n综测成绩计算公式..."

    monkeypatch.setattr("services.retrieval.get_rag_context_async", fake_rag_context)
    result = await collect_evidence(
        "我的个人综测成绩是多少？",
        retriever=object(),
        tools=FakeToolRegistry([]),
        web_search_enabled=False,
    )
    assert called["rag"] == 0
    assert result.decision == "out_of_scope"
    assert result.sources == []
    assert result.knowledge_context == ""


@pytest.mark.asyncio
async def test_collect_evidence_hides_weak_local_sources_for_an_out_of_scope_query(monkeypatch) -> None:
    async def fake_rag_context(query, retriever):
        return "【学生手册｜第 12 页】\n国家奖学金奖励标准为 10000 元。"

    async def fake_probe(query, retriever, context):
        # Very weak vector + no overlap => out_of_scope (BM25 alone cannot rescue).
        return context, 0.02, 5.71, 0.0

    monkeypatch.setattr("services.retrieval.get_rag_context_async", fake_rag_context)
    monkeypatch.setattr("services.retrieval._probe_local_retrieval", fake_probe)

    result = await collect_evidence("图书馆今天有座位吗", retriever=object(), tools=None, web_search_enabled=False)

    assert result.decision == "out_of_scope"
    assert result.sources == []
