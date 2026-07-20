"""Deterministic confidence scoring for cited answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from services.evidence import EvidenceSource


ConfidenceLevel = Literal["high", "medium", "low", "unknown"]
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


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

    uncertain_points: list[str] = []
    if _has_numeric_conflict(evidence):
        score -= 0.55
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

    knowledge_count = sum(source.source_type == "knowledge_base" for source in evidence)
    web_count = len(evidence) - knowledge_count
    summary_parts = []
    if knowledge_count:
        summary_parts.append(f"{knowledge_count} 条校内知识库资料")
    if web_count:
        summary_parts.append(f"{web_count} 条网络资料")
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


def _has_numeric_conflict(sources: list[EvidenceSource]) -> bool:
    numeric_sets = {
        frozenset(_NUMBER_PATTERN.findall(source.snippet))
        for source in sources
        if _NUMBER_PATTERN.findall(source.snippet)
    }
    return len(numeric_sets) > 1
