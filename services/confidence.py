"""Deterministic confidence scoring for cited answers."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal

from services.evidence import EvidenceSource


ConfidenceLevel = Literal["high", "medium", "low", "unknown"]

# Numbers with a short left-context + unit — used for real claim conflicts.
# Avoid treating every digit on neighboring statute pages as one global set.
_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?P<label>[\u4e00-\u9fffA-Za-z]{1,16}?)(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>分/人次|分|元|%|％)"
    ),
    re.compile(
        r"(?P<label>奖励标准为|标准为|金额为|为)\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>元)?"
    ),
)

# Labels that are too generic to compare across pages.
_GENERIC_LABELS = {
    "为",
    "的",
    "和",
    "与",
    "及",
    "中",
    "第",
    "共",
    "约",
    "达",
    "按",
    "另",
    "各",
    "每",
    "共有",
}


@dataclass(frozen=True)
class ConfidenceResult:
    score: float
    level: ConfidenceLevel
    evidence_summary: str
    uncertain_points: list[str]


def calculate_confidence(sources: Iterable[EvidenceSource]) -> ConfidenceResult:
    evidence = list(sources)
    if not evidence:
        return ConfidenceResult(
            score=0.0,
            level="unknown",
            evidence_summary="未检索到可核验的参考资料。",
            uncertain_points=["当前回答缺少可核验的来源。"],
        )

    best = max(
        source.relevance_score * 0.5 + source.authority_score * 0.5
        for source in evidence
    )
    score = best
    if len(evidence) >= 2:
        score += 0.05
    if any(_has_complete_metadata(source) for source in evidence):
        score += 0.05
    # Pure on-campus statute citations are usually more trustworthy than thin web hits.
    if any(source.source_type == "knowledge_base" for source in evidence):
        score += 0.08

    uncertain_points: list[str] = []
    if _has_numeric_conflict(evidence):
        score -= 0.35
        uncertain_points.append("不同来源中的数值信息存在冲突，请以校内正式文件为准。")
    elif (
        not any(source.source_type == "knowledge_base" for source in evidence)
        and any(source.source_type == "web" and not source.published_at for source in evidence)
    ):
        uncertain_points.append("部分网页未提供发布日期，时效性需要进一步核验。")

    bounded_score = round(max(0.0, min(1.0, score)), 2)
    if bounded_score >= 0.8 and not uncertain_points:
        level: ConfidenceLevel = "high"
    elif bounded_score >= 0.55:
        level = "medium"
    elif bounded_score >= 0.3:
        level = "low"
    else:
        level = "unknown"

    # Never show "unknown" when we actually have complete campus citations and
    # no explicit conflict — mid vector scores are common for colloquial asks.
    if (
        level == "unknown"
        and not uncertain_points
        and any(
            source.source_type == "knowledge_base" and _has_complete_metadata(source)
            for source in evidence
        )
    ):
        level = "medium"
        bounded_score = max(bounded_score, 0.55)

    knowledge_count = sum(source.source_type == "knowledge_base" for source in evidence)
    web_count = len(evidence) - knowledge_count
    summary_parts = []
    if knowledge_count:
        summary_parts.append(f"{knowledge_count}条校内知识库资料")
    if web_count:
        summary_parts.append(f"{web_count}条网络资料")
    return ConfidenceResult(
        score=bounded_score,
        level=level,
        evidence_summary=f"依据{'、'.join(summary_parts)}生成。",
        uncertain_points=uncertain_points,
    )


def _has_complete_metadata(source: EvidenceSource) -> bool:
    if source.source_type == "web":
        return bool(source.url and source.site_name)
    return bool((source.document_name or source.document_id) and source.page is not None)


def _normalize_label(label: str) -> str:
    text = re.sub(r"\s+", "", label or "")
    # Drop trailing particles that do not identify the metric.
    text = re.sub(r"[的之了其]$", "", text)
    # Keep the most specific tail of the label (e.g. 晚归 / 国家奖学金奖励标准).
    if len(text) > 12:
        text = text[-12:]
    return text


_CLAUSE_SPLIT_RE = re.compile(r"[。；：;]")
_QUOTED_RE = re.compile(r"[“\"「『]([^”\"」』]{1,12})[”\"」』]")


def _expand_label_with_topic(text: str, start: int, label: str) -> str:
    """Pull a short topic word immediately before a thin label like 扣/为/加.

    Production rules:
    - Scope attribution to the CURRENT clause (split on 。；：) so a topic from
      a neighbouring penalty/reward row never bleeds into this claim.
    - Keep quoted phrases (e.g. “卫生寝室”) in the fallback key; otherwise
      distinct clauses like “卫生寝室”荣誉…加5分 / “五星文明寝室”荣誉…加10分
      collapse into one metric and look like a numeric conflict.
    """
    base = _normalize_label(label)
    window = re.sub(r"\s+", "", text[max(0, start - 24) : start])
    clause_left = _CLAUSE_SPLIT_RE.split(window)[-1] if window else ""
    clause = f"{clause_left}{base}"
    quoted = _QUOTED_RE.findall(clause)
    if base and base not in {"扣", "为", "达", "共", "按"} and len(base) >= 2:
        # Long label (usually a full CJK run). Disambiguate with the nearest
        # quoted phrase in the same clause, otherwise rows like
        # “卫生寝室”荣誉…加5分 / “五星文明寝室”荣誉…加10分 collapse into one key.
        if quoted and quoted[-1] not in base:
            return f"{quoted[-1]}{base}"
        return base
    # Prefer known campus metric heads if present in the current clause.
    for topic in (
        "夜不归宿",
        "晚归",
        "国家励志奖学金",
        "国家奖学金",
        "国家助学金",
        "学业奖学金",
        "奖励标准",
    ):
        if topic in clause_left:
            return topic if not base or base in {"扣", "为"} else f"{topic}{base}"
    # Fallback: nearest quoted phrase + last CJK chars of the clause.
    chars = re.findall(r"[\u4e00-\u9fff]+", clause_left)
    if chars:
        tail = chars[-1][-6:]
        if base and base in tail:
            combined = tail
        elif base:
            combined = f"{tail}{base}"
        else:
            combined = tail
        if quoted and quoted[-1] not in combined:
            return f"{quoted[-1]}{combined}"
        return combined
    if quoted:
        return quoted[-1]
    return base


def _extract_metric_claims(snippet: str) -> list[tuple[str, str]]:
    """Return (metric_key, value) pairs from a snippet."""
    text = snippet or ""
    claims: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pattern in _CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            raw_label = match.group("label") or ""
            label = _expand_label_with_topic(text, match.start("label"), raw_label)
            value = match.group("value")
            unit = (match.groupdict().get("unit") or "").strip()
            if not value:
                continue
            if label in _GENERIC_LABELS and unit not in {"元", "分", "分/人次", "%", "％"}:
                continue
            # Metric key prefers label+unit so 晚归…5分 ≠ 夜不归宿…20分.
            key = f"{label}|{unit}" if label else f"|{unit}"
            item = (key, value)
            if item in seen:
                continue
            seen.add(item)
            claims.append(item)
    return claims


def _same_campus_document(sources: list[EvidenceSource]) -> bool:
    """True when all KB hits are from one document family (neighbor pages etc.)."""
    kb = [source for source in sources if source.source_type == "knowledge_base"]
    if len(kb) < 2 or len(kb) != len(sources):
        return False
    ids = {source.document_id for source in kb if source.document_id}
    names = {source.document_name for source in kb if source.document_name}
    return len(ids) <= 1 and len(names) <= 1


def _has_numeric_conflict(sources: list[EvidenceSource]) -> bool:
    """Flag only real claim conflicts, not unrelated numbers on adjacent pages.

    Examples that SHOULD conflict:
      - 国家奖学金 10000 元 vs 8000 元
    Examples that should NOT:
      - p145 晚归扣5分 and 夜不归宿扣20分
      - p138 门禁 23:30 and p145 扣5分
      - same handbook neighbor pages with different clause numbers
    """
    if len(sources) < 2:
        return False

    # Neighbor pages from the same campus manual almost always contain many
    # unrelated numbers (article nos, times, different penalty rows). Treat them
    # as non-conflicting unless the same metric label disagrees.
    values_by_metric: dict[str, set[str]] = defaultdict(set)
    for source in sources:
        for metric, value in _extract_metric_claims(source.snippet):
            values_by_metric[metric].add(value)

    real_conflicts = {
        metric: values
        for metric, values in values_by_metric.items()
        if len(values) > 1 and not metric.startswith("|")  # require some label
    }
    if real_conflicts:
        return True

    # Fallback legacy behavior only when comparing mixed/foreign sources and we
    # could not parse labeled claims — still avoid same-document KB neighbors.
    if _same_campus_document(sources):
        return False

    numeric_sets = {
        frozenset(re.findall(r"\d+(?:\.\d+)?", source.snippet or ""))
        for source in sources
        if re.findall(r"\d+(?:\.\d+)?", source.snippet or "")
    }
    # Only conflict when two sources share at least one number family poorly —
    # i.e. both look like money/award statements. Otherwise skip.
    moneyish = [
        source
        for source in sources
        if re.search(r"\d+\s*元|奖励标准|奖学金", source.snippet or "")
    ]
    if len(moneyish) >= 2:
        money_sets = {
            frozenset(re.findall(r"\d+(?:\.\d+)?", source.snippet or ""))
            for source in moneyish
        }
        return len(money_sets) > 1

    return False
