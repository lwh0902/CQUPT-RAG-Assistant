"""Dual retrieval orchestration for local RAG evidence and Agent web tools."""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from typing import Any

from rag import extract_query_keywords, get_rag_context_async, init_bm25_index
from settings import (
    RETRIEVAL_BM25_SUPPORTED_THRESHOLD,
    RETRIEVAL_BM25_VECTOR_FLOOR,
    RETRIEVAL_KEYWORD_COVERAGE_THRESHOLD,
    RETRIEVAL_VECTOR_OUT_OF_SCOPE_THRESHOLD,
    RETRIEVAL_VECTOR_SUPPORTED_THRESHOLD,
)
from services.evidence import EvidenceSource, deduplicate_evidence, normalize_evidence


_CONTEXT_PATTERN = re.compile(
    r"【(?P<document>.+?)(?:｜第\s*(?P<page>\d+)\s*页)?】\s*\n?(?P<snippet>.*?)(?=\n\n【|\Z)",
    re.S,
)

# Personal-record questions cannot be answered from policy manuals alone.
_PERSONAL_DATA_QUERY_RE = re.compile(
    r"("
    r"我的(个人)?(综测|综合测评|成绩|学分|绩点|GPA|学号|宿舍|寝室|班级)|"
    r"(查|查询|看看)?(一下)?我的(综测|综合测评|成绩|学分|绩点|GPA)|"
    r"个人(综测|综合测评)?成绩是多少"
    r")",
    re.I,
)


def is_personal_data_query(query: str) -> bool:
    """True when the user asks for private/personal records, not policy text."""
    text = (query or "").strip()
    if not text:
        return False
    return bool(_PERSONAL_DATA_QUERY_RE.search(text))


# Dynamic / entity-specific asks a static policy manual can never answer
# (live status, phone numbers, specific person/course/office). Same design
# class as is_personal_data_query: stable ask-TYPE patterns, not topic routers.
_DYNAMIC_HARD_MARKERS = (
    "外卖", "菜单", "座位", "密码", "报修", "校车", "快递", "有号",
)
_DYNAMIC_SOFT_MARKERS = (
    "今天", "今日", "明天", "今晚", "本周", "这周", "当前", "现在", "实时",
    "几点", "营业时间", "电话", "联系方式", "在办公室", "价格", "哪家好",
    "具体时间", "具体日期", "上课地点",
)
# Rule-intent words rescue a soft-dynamic phrasing back to statute QA
# (e.g. "奖学金什么时候申请", "补考具体时间规定").
_RULE_INTENT_MARKERS = (
    "规定", "办法", "条件", "资格", "标准", "流程", "怎么办", "怎么算",
    "种类", "处分", "扣分", "评定", "申请", "计算", "休学", "复学",
    "退学", "转专业", "补考", "重修",
)


def is_dynamic_info_query(query: str) -> bool:
    """True for live-status / entity-specific asks the manual cannot answer.

    Hard markers (外卖/菜单/报修...) always fire. Soft markers (今天/几点/电话...)
    fire only when the question has no rule intent and hits no covered policy
    topic (alias lexicon) — so "宿舍门禁几点关门" still reaches the manual.
    """
    text = " ".join((query or "").strip().split())
    if not text:
        return False
    if any(marker in text for marker in _DYNAMIC_HARD_MARKERS):
        return True
    if not any(marker in text for marker in _DYNAMIC_SOFT_MARKERS):
        return False
    if any(marker in text for marker in _RULE_INTENT_MARKERS):
        return False
    # Covered-topic rescue via the existing policy alias lexicon.
    from services.query_normalize import _alias_hits

    if _alias_hits(text):
        return False
    return True


@dataclass(frozen=True)
class RetrievedEvidence:
    knowledge_context: str
    sources: list[EvidenceSource]
    web_search_enabled: bool
    decision: str
    vector_score: float | None
    bm25_score: float
    keyword_coverage: float

    @property
    def web_sources(self) -> list[EvidenceSource]:
        return [source for source in self.sources if source.source_type == "web"]


async def collect_evidence(
    query: str,
    *,
    retriever: Any,
    tools: Any,
    web_search_enabled: bool,
) -> RetrievedEvidence:
    # Personal records are never in the mounted policy corpus.
    if is_personal_data_query(query) or is_dynamic_info_query(query):
        web_sources = []
        if web_search_enabled and tools is not None:
            web_sources = await tools.search_web(query)
        if web_sources:
            return RetrievedEvidence(
                knowledge_context="",
                sources=deduplicate_evidence(web_sources),
                web_search_enabled=web_search_enabled,
                decision="web_only",
                vector_score=None,
                bm25_score=0.0,
                keyword_coverage=0.0,
            )
        return RetrievedEvidence(
            knowledge_context="",
            sources=[],
            web_search_enabled=web_search_enabled,
            decision="out_of_scope",
            vector_score=None,
            bm25_score=0.0,
            keyword_coverage=0.0,
        )

    async def get_local_context() -> tuple[str, float | None, float, float]:
        if retriever is None:
            return "", None, 0.0, 0.0
        context = await get_rag_context_async(query, retriever)
        return await _probe_local_retrieval(query, retriever, context)

    local_task = get_local_context()
    web_task = tools.search_web(query) if web_search_enabled and tools is not None else _empty_sources()
    (knowledge_context, vector_score, bm25_score, keyword_coverage), web_sources = await asyncio.gather(
        local_task,
        web_task,
    )
    local_sources = knowledge_context_to_evidence(knowledge_context, relevance_score=vector_score)
    local_decision = decide_local_evidence(
        has_local_source=bool(local_sources),
        vector_score=vector_score,
        bm25_score=bm25_score,
        keyword_coverage=keyword_coverage,
    )
    if local_decision == "supported":
        decision = "supported"
        sources = deduplicate_evidence([*local_sources, *web_sources])
    elif web_sources:
        decision = "web_only"
        sources = deduplicate_evidence(web_sources)
    else:
        decision = local_decision
        sources = []
    return RetrievedEvidence(
        knowledge_context=knowledge_context,
        sources=sources,
        web_search_enabled=web_search_enabled,
        decision=decision,
        vector_score=vector_score,
        bm25_score=bm25_score,
        keyword_coverage=keyword_coverage,
    )


def decide_local_evidence(
    *,
    has_local_source: bool,
    vector_score: float | None,
    bm25_score: float,
    keyword_coverage: float,
) -> str:
    """Classify local retrieval without treating Top-K presence as evidence.

    Supported paths (any one is enough):
    1) strong vector similarity
    2) solid keyword overlap with non-zero lexical score AND vector above a
       floor — campus-flavored but unanswerable asks (hours/phone/menu) get
       high BM25 + coverage from generic topic words, so BM25 alone must not
       override a clearly weak vector match
    3) strong BM25 plus not-out-of-scope vector (covers short colloquial asks
       whose wording differs from statute text but still hits the right pages)
    """
    if not has_local_source:
        return "out_of_scope"
    if vector_score is None:
        # Older/custom retrievers may not expose relevance scores. Keep the
        # existing behavior instead of creating a false negative.
        return "supported"
    # Very weak semantic match + no keyword overlap: treat as out of corpus.
    # (Strong BM25 alone is not enough here — it can fire on generic campus words.)
    if (
        vector_score < RETRIEVAL_VECTOR_OUT_OF_SCOPE_THRESHOLD
        and keyword_coverage == 0
    ):
        return "out_of_scope"
    if vector_score >= RETRIEVAL_VECTOR_SUPPORTED_THRESHOLD:
        return "supported"
    if (
        bm25_score > 0
        and keyword_coverage >= RETRIEVAL_KEYWORD_COVERAGE_THRESHOLD
        and vector_score >= RETRIEVAL_BM25_VECTOR_FLOOR
    ):
        return "supported"
    # Mid vector + any real overlap is usually enough for statute QA.
    if (
        vector_score >= RETRIEVAL_VECTOR_OUT_OF_SCOPE_THRESHOLD
        and keyword_coverage > 0
        and bm25_score > 0
    ):
        return "supported"
    # Mid vector + strong lexical hit, even if closed-lexicon coverage is thin.
    if (
        vector_score >= RETRIEVAL_VECTOR_OUT_OF_SCOPE_THRESHOLD
        and bm25_score >= RETRIEVAL_BM25_SUPPORTED_THRESHOLD
    ):
        return "supported"
    return "insufficient"


def should_rewrite_after_base_retrieval(
    *,
    vector_score: float | None,
    bm25_score: float,
    keyword_coverage: float,
) -> bool:
    """Decide whether a knowledge query should run gated rewrite after base hybrid.

    Policy:
    - Strong base hit: skip rewrite (protect precise/formal queries).
    - Clear out-of-scope: skip rewrite (rewrite rarely helps and costs latency).
    - Weak in-between: rewrite once to bridge colloquial gaps.
    - Missing vector score: skip rewrite to avoid harming precise queries when
      probes are unavailable.
    """
    if vector_score is None:
        return False
    decision = decide_local_evidence(
        has_local_source=True,
        vector_score=vector_score,
        bm25_score=bm25_score,
        keyword_coverage=keyword_coverage,
    )
    if decision == "out_of_scope":
        return False
    if decision == "supported":
        return False
    # Weak but not clearly out-of-scope: allow one rewrite pass.
    return True


def probe_retrieval_scores(
    query: str,
    retriever: Any,
    docs: list[Any] | None = None,
) -> tuple[float | None, float, float]:
    """Return (vector_score, bm25_score, keyword_coverage) for a query."""
    context = ""
    if docs:
        from rag import format_context

        context, _ = format_context(docs)
    vector_score = _top_vector_score(query, retriever)
    bm25_score = _top_bm25_score(query)
    keyword_coverage = _keyword_coverage(query, context)
    return vector_score, bm25_score, keyword_coverage


def decide_from_docs(query: str, retriever: Any, docs: list[Any]) -> str:
    """Evidence-gate decision for retrieved docs (used by evaluation)."""
    if is_personal_data_query(query) or is_dynamic_info_query(query):
        return "out_of_scope"
    if not docs:
        return "out_of_scope"
    vector_score, bm25_score, keyword_coverage = probe_retrieval_scores(query, retriever, docs)
    return decide_local_evidence(
        has_local_source=True,
        vector_score=vector_score,
        bm25_score=bm25_score,
        keyword_coverage=keyword_coverage,
    )


async def _probe_local_retrieval(
    query: str,
    retriever: Any,
    context: str,
) -> tuple[str, float | None, float, float]:
    return await asyncio.to_thread(_probe_local_retrieval_sync, query, retriever, context)


def _probe_local_retrieval_sync(
    query: str,
    retriever: Any,
    context: str,
) -> tuple[str, float | None, float, float]:
    vector_score = _top_vector_score(query, retriever)
    bm25_score = _top_bm25_score(query)
    keyword_coverage = _keyword_coverage(query, context)
    return context, vector_score, bm25_score, keyword_coverage


def _top_vector_score(query: str, retriever: Any) -> float | None:
    vector_store = getattr(retriever, "vectorstore", None)
    search = getattr(vector_store, "similarity_search_with_relevance_scores", None)
    if not callable(search):
        return None
    try:
        results = search(query, k=1)
    except Exception:
        return None
    if not results:
        return 0.0
    try:
        return max(0.0, min(1.0, float(results[0][1])))
    except (IndexError, TypeError, ValueError):
        return None


def _top_bm25_score(query: str) -> float:
    try:
        results = init_bm25_index().search(query, top_k=1)
        return max(0.0, float(results[0][1])) if results else 0.0
    except Exception:
        return 0.0


def _keyword_coverage(query: str, context: str) -> float:
    keywords = extract_query_keywords(query)
    if not keywords:
        return 0.0
    matched = sum(1 for keyword in keywords if keyword in context)
    return matched / len(keywords)


def knowledge_context_to_evidence(
    context: str,
    *,
    relevance_score: float | None = None,
) -> list[EvidenceSource]:
    evidence: list[EvidenceSource] = []
    for match in _CONTEXT_PATTERN.finditer(context or ""):
        document_name = match.group("document").strip()
        snippet = " ".join(match.group("snippet").split())
        if not document_name or not snippet:
            continue
        page_value = match.group("page")
        source = normalize_evidence(
            {
                "id": f"kb:{hashlib.sha256(f'{document_name}:{page_value}:{snippet}'.encode('utf-8')).hexdigest()[:16]}",
                "source_type": "knowledge_base",
                "title": document_name,
                "document_name": document_name,
                "page": int(page_value) if page_value else None,
                "snippet": snippet,
                "relevance_score": relevance_score if relevance_score is not None else 0.0,
                "authority_score": 0.95,
            }
        )
        if source is not None:
            evidence.append(source)
    return evidence


def format_web_evidence_for_prompt(sources: list[EvidenceSource]) -> str:
    if not sources:
        return "无"
    return "\n\n".join(
        f"【{source.title}｜{source.url}】\n{source.snippet}"
        for source in sources
    )


async def _empty_sources() -> list[EvidenceSource]:
    return []
