"""Evidence normalization shared by local retrieval and web search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional
from urllib.parse import urlsplit, urlunsplit


SourceType = Literal["knowledge_base", "web"]


def _bounded_score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def canonicalize_url(value: str) -> Optional[str]:
    """Return a stable HTTP(S) URL, or None for unverified web references."""
    parsed = urlsplit((value or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


@dataclass(frozen=True)
class EvidenceSource:
    id: str
    source_type: SourceType
    title: str
    snippet: str
    url: Optional[str] = None
    site_name: Optional[str] = None
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    page: Optional[int] = None
    published_at: Optional[str] = None
    retrieved_at: Optional[str] = None
    relevance_score: float = 0.0
    authority_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_type": self.source_type,
            "title": self.title,
            "url": self.url,
            "site_name": self.site_name,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page": self.page,
            "snippet": self.snippet,
            "published_at": self.published_at,
            "retrieved_at": self.retrieved_at,
            "relevance_score": self.relevance_score,
            "authority_score": self.authority_score,
        }


def normalize_evidence(raw: dict[str, Any]) -> Optional[EvidenceSource]:
    source_type = raw.get("source_type")
    title = str(raw.get("title") or "").strip()
    snippet = str(raw.get("snippet") or "").strip()
    source_id = str(raw.get("id") or "").strip()
    if source_type not in {"knowledge_base", "web"} or not title or not snippet or not source_id:
        return None

    if source_type == "web":
        url = canonicalize_url(str(raw.get("url") or ""))
        if url is None:
            return None
        return EvidenceSource(
            id=source_id,
            source_type="web",
            title=title,
            snippet=snippet,
            url=url,
            site_name=_clean_optional(raw.get("site_name")),
            published_at=_clean_optional(raw.get("published_at")),
            retrieved_at=_clean_optional(raw.get("retrieved_at")),
            relevance_score=_bounded_score(raw.get("relevance_score")),
            authority_score=_bounded_score(raw.get("authority_score")),
        )

    document_id = _clean_optional(raw.get("document_id"))
    document_name = _clean_optional(raw.get("document_name"))
    if not document_id and not document_name:
        return None

    page = raw.get("page")
    try:
        normalized_page = int(page) if page is not None else None
    except (TypeError, ValueError):
        normalized_page = None

    return EvidenceSource(
        id=source_id,
        source_type="knowledge_base",
        title=title,
        snippet=snippet,
        document_id=document_id,
        document_name=document_name,
        page=normalized_page,
        published_at=_clean_optional(raw.get("published_at")),
        retrieved_at=_clean_optional(raw.get("retrieved_at")),
        relevance_score=_bounded_score(raw.get("relevance_score")),
        authority_score=_bounded_score(raw.get("authority_score")),
    )


def deduplicate_evidence(sources: Iterable[EvidenceSource]) -> list[EvidenceSource]:
    strongest_by_key: dict[str, EvidenceSource] = {}
    for source in sources:
        if source.source_type == "web":
            key = f"web:{canonicalize_url(source.url or '') or source.id}"
        else:
            document_key = source.document_id or source.document_name or source.id
            key = f"knowledge_base:{document_key}:{source.page or 0}"

        current = strongest_by_key.get(key)
        if current is None or _rank(source) > _rank(current):
            strongest_by_key[key] = source

    return sorted(strongest_by_key.values(), key=_rank, reverse=True)


def _clean_optional(value: Any) -> Optional[str]:
    normalized = str(value or "").strip()
    return normalized or None


def _rank(source: EvidenceSource) -> tuple[float, float, int]:
    return (
        source.authority_score,
        source.relevance_score,
        1 if source.source_type == "knowledge_base" else 0,
    )
