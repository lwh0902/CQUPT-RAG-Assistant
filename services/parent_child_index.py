"""Build child retrieval docs and parent context map from policy pages."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from services.article_chunker import merge_articles_into_parents, parse_articles_from_pages

PARENT_STORE_PATH = Path("parent_chunks.json")
_ARTICLE_MARK_RE = __import__("re").compile(r"第\s*[零〇一二三四五六七八九十百千0-9]+\s*条")


def _meta_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_for_coverage(text: str) -> str:
    return "".join((text or "").split())


def _make_child_document(
    *,
    doc_id: str,
    doc_name: str,
    extra: dict[str, Any],
    text: str,
    page_start: int,
    page_end: int,
    parent_id: str,
    child_id: str,
    article_no: int = 0,
    article_title: str = "",
    chapter: str = "",
    section: str = "",
    header_bits: list[str] | None = None,
) -> Document:
    bits = header_bits or [doc_name]
    header = "｜".join([b for b in bits if b])
    content = f"{header}\n{text}".strip() if header else text
    metadata = {
        "document_id": doc_id,
        "document_name": doc_name,
        "document_type": _meta_str(extra.get("document_type")),
        "topic": _meta_str(extra.get("topic")),
        "authority_level": int(extra.get("authority_level") or 0),
        "source": _meta_str(extra.get("source")),
        "file_path": _meta_str(extra.get("file_path")),
        "page": int(page_start) if page_start else 0,
        "page_end": int(page_end) if page_end else 0,
        "parent_id": parent_id,
        "child_id": child_id,
        "article_no": int(article_no) if isinstance(article_no, int) else 0,
        "article_title": article_title,
        "chapter": chapter,
        "section": section,
        "chunk_type": "child",
    }
    return Document(page_content=content, metadata=metadata)


def _add_page_level_children(
    *,
    pages: list[dict[str, Any]],
    doc_id: str,
    doc_name: str,
    extra: dict[str, Any],
    child_docs: list[Document],
    parent_store: dict[str, dict[str, Any]],
    min_page_chars: int = 20,
) -> None:
    """Always index full pages as an additional retrieval lane.

    Why both article children and page children:
    - article children: precise clause QA / colloquial policy questions
    - page children: keep coverage for page-seed / long excerpt queries and
      front matter without 第X条 markers
    """
    for page_item in pages:
        page_no = int(page_item.get("page") or 0)
        raw = (page_item.get("text") or "").strip()
        compact_len = len("".join(raw.split()))
        if page_no <= 0 or compact_len < min_page_chars:
            continue

        parent_id = f"{doc_id}:page:{page_no}"
        # Page parent is the page itself (not merged articles).
        parent_store[parent_id] = {
            "parent_id": parent_id,
            "document_id": doc_id,
            "document_name": doc_name,
            "title": f"第{page_no}页",
            "text": raw,
            "page_start": page_no,
            "page_end": page_no,
            "article_nos": [],
            "merge_count": 1,
            "document_type": extra.get("document_type"),
            "topic": extra.get("topic"),
            "authority_level": extra.get("authority_level"),
            "source": extra.get("source"),
            "file_path": extra.get("file_path"),
            "chunk_origin": "page",
        }
        child_docs.append(
            _make_child_document(
                doc_id=doc_id,
                doc_name=doc_name,
                extra=extra,
                text=raw,
                page_start=page_no,
                page_end=page_no,
                parent_id=parent_id,
                child_id=f"{parent_id}:full",
                article_title=f"第{page_no}页",
                header_bits=[doc_name, f"第{page_no}页"],
            )
        )


def build_parent_child_corpus(
    page_documents: list[Document],
    *,
    min_parent_chars: int = 220,
    max_parent_chars: int = 900,
) -> tuple[list[Document], dict[str, dict[str, Any]]]:
    """Return (child Documents for indexing, parent_id -> parent payload)."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    names: dict[str, str] = {}
    extras: dict[str, dict[str, Any]] = {}

    for doc in page_documents:
        doc_id = _meta_str(doc.metadata.get("document_id") or "unknown")
        names[doc_id] = _meta_str(doc.metadata.get("document_name") or doc_id)
        extras[doc_id] = {
            "document_type": doc.metadata.get("document_type"),
            "topic": doc.metadata.get("topic"),
            "authority_level": doc.metadata.get("authority_level"),
            "source": doc.metadata.get("source"),
            "file_path": doc.metadata.get("file_path"),
        }
        page = doc.metadata.get("page")
        try:
            page_no = int(page)
        except (TypeError, ValueError):
            page_no = 0
        grouped[doc_id].append({"page": page_no, "text": doc.page_content or ""})

    child_docs: list[Document] = []
    parent_store: dict[str, dict[str, Any]] = {}

    for doc_id, pages in grouped.items():
        pages.sort(key=lambda item: item["page"])
        articles = parse_articles_from_pages(
            pages,
            document_id=doc_id,
            document_name=names[doc_id],
        )
        parents, children = merge_articles_into_parents(
            articles,
            min_parent_chars=min_parent_chars,
            max_parent_chars=max_parent_chars,
        )
        extra = extras.get(doc_id, {})
        for parent in parents:
            parent_id = _meta_str(parent.get("parent_id"))
            parent_store[parent_id] = {
                **parent,
                "document_type": extra.get("document_type"),
                "topic": extra.get("topic"),
                "authority_level": extra.get("authority_level"),
                "source": extra.get("source"),
                "file_path": extra.get("file_path"),
                "chunk_origin": "article",
            }
        for child in children:
            parent_id = _meta_str(child.get("parent_id"))
            page_start = child.get("page_start") or 0
            page_end = child.get("page_end") or page_start
            text = (child.get("text") or "").strip()
            if not text:
                continue
            title = _meta_str(child.get("article_title"))
            article_no = child.get("article_no")
            header_bits = [names[doc_id]]
            if article_no is not None:
                header_bits.append(f"第{article_no}条")
            if title:
                header_bits.append(title)
            child_docs.append(
                _make_child_document(
                    doc_id=doc_id,
                    doc_name=names[doc_id],
                    extra=extra,
                    text=text,
                    page_start=int(page_start) if page_start else 0,
                    page_end=int(page_end) if page_end else 0,
                    parent_id=parent_id,
                    child_id=_meta_str(child.get("child_id")),
                    article_no=int(article_no) if isinstance(article_no, int) else 0,
                    article_title=title,
                    chapter=_meta_str(child.get("chapter")),
                    section=_meta_str(child.get("section")),
                    header_bits=header_bits,
                )
            )

        # Dual lane: full pages keep page-seed coverage; articles keep clause precision.
        _add_page_level_children(
            pages=pages,
            doc_id=doc_id,
            doc_name=names[doc_id],
            extra=extra,
            child_docs=child_docs,
            parent_store=parent_store,
        )

    return child_docs, parent_store


def save_parent_store(parent_store: dict[str, dict[str, Any]], path: Path = PARENT_STORE_PATH) -> None:
    path.write_text(json.dumps(parent_store, ensure_ascii=False, indent=2), encoding="utf-8")


def load_parent_store(path: Path = PARENT_STORE_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parent_to_document(parent: dict[str, Any], *, page: int | None = None) -> Document:
    """Build a parent context Document from a parent_store payload."""
    parent_id = _meta_str(parent.get("parent_id"))
    page_start = int(parent.get("page_start") or 0)
    page_end = int(parent.get("page_end") or page_start)
    cite_page = int(page) if page is not None else page_start
    article_nos = parent.get("article_nos") or []
    title = _meta_str(parent.get("title"))
    text = _meta_str(parent.get("text"))
    header = _meta_str(parent.get("document_name"))
    if title:
        header = f"{header}｜{title}"
    content = f"{header}\n{text}".strip()
    metadata = {
        "document_id": _meta_str(parent.get("document_id")),
        "document_name": _meta_str(parent.get("document_name")),
        "document_type": _meta_str(parent.get("document_type")),
        "topic": _meta_str(parent.get("topic")),
        "authority_level": int(parent.get("authority_level") or 0),
        "source": _meta_str(parent.get("source")),
        "file_path": _meta_str(parent.get("file_path")),
        "page": cite_page,
        "page_end": page_end,
        "parent_id": parent_id,
        "article_nos": ",".join(str(x) for x in article_nos),
        "article_title": title,
        "chapter": _meta_str(parent.get("chapter")),
        "section": _meta_str(parent.get("section")),
        "chunk_type": "parent",
        "merge_count": int(parent.get("merge_count") or 1),
        "chunk_origin": _meta_str(parent.get("chunk_origin") or "page"),
        "neighbor_expanded": True,
    }
    return Document(page_content=content, metadata=metadata)


def expand_neighbor_pages(
    docs: list[Document],
    parent_store: dict[str, dict[str, Any]] | None = None,
    *,
    radius: int = 1,
    max_seed_docs: int = 5,
) -> list[Document]:
    """Append same-document neighbor pages (page±radius) for top hits.

    Production goal: when a clause spans OCR page breaks, keep adjacent page
    text available for answer completeness without another full retrieval.
    Seeds are the current ranked hits; neighbors are appended after seeds so
    original rank order is preserved until a later rerank step.
    """
    if not docs or radius <= 0:
        return list(docs)
    store = parent_store if parent_store is not None else load_parent_store()
    if not store:
        return list(docs)

    output = list(docs)
    seen_pages: set[tuple[str, int]] = set()
    for doc in docs:
        key = _page_dedupe_key(doc)
        if key is not None:
            seen_pages.add(key)

    for seed in docs[: max(0, max_seed_docs)]:
        meta = getattr(seed, "metadata", None) or {}
        doc_id = _meta_str(meta.get("document_id"))
        try:
            page_no = int(meta.get("page") or 0)
        except (TypeError, ValueError):
            page_no = 0
        if not doc_id or page_no <= 0:
            continue
        for delta in range(-radius, radius + 1):
            if delta == 0:
                continue
            neighbor_page = page_no + delta
            if neighbor_page <= 0:
                continue
            page_key = (doc_id, neighbor_page)
            if page_key in seen_pages:
                continue
            parent_id = f"{doc_id}:page:{neighbor_page}"
            parent = store.get(parent_id)
            if not parent:
                continue
            seen_pages.add(page_key)
            output.append(_parent_to_document(parent, page=neighbor_page))
    return output


def _page_dedupe_key(doc: Document) -> tuple[str, int] | None:
    meta = getattr(doc, "metadata", None) or {}
    doc_id = _meta_str(meta.get("document_id"))
    page = meta.get("page")
    try:
        page_no = int(page)
    except (TypeError, ValueError):
        return None
    if not doc_id or page_no <= 0:
        return None
    return doc_id, page_no


def expand_children_to_parents(
    child_docs: list[Document],
    parent_store: dict[str, dict[str, Any]] | None = None,
) -> list[Document]:
    """Map ranked child hits to unique parent context documents.

    Dedup rules (preserve rank order):
    1. same parent_id once
    2. same (document_id, page) once — dual-lane article+page hits often land on
       the same citation page and would otherwise waste Top-K slots
    """
    store = parent_store if parent_store is not None else load_parent_store()
    output: list[Document] = []
    seen_parent: set[str] = set()
    seen_page: set[tuple[str, int]] = set()

    def _append(doc: Document) -> None:
        page_key = _page_dedupe_key(doc)
        if page_key is not None and page_key in seen_page:
            return
        if page_key is not None:
            seen_page.add(page_key)
        output.append(doc)

    for child in child_docs:
        parent_id = _meta_str((child.metadata or {}).get("parent_id"))
        if parent_id and parent_id in store:
            if parent_id in seen_parent:
                continue
            seen_parent.add(parent_id)
            parent = store[parent_id]
            # Keep the hit child's page for citation/eval; parent text for context.
            child_page = child.metadata.get("page") or parent.get("page_start") or 0
            page_end = parent.get("page_end") or child.metadata.get("page_end") or child_page
            article_nos = parent.get("article_nos") or []
            title = _meta_str(parent.get("title"))
            text = _meta_str(parent.get("text"))
            header = _meta_str(parent.get("document_name"))
            if title:
                header = f"{header}｜{title}"
            content = f"{header}\n{text}".strip()
            metadata = {
                "document_id": _meta_str(parent.get("document_id") or child.metadata.get("document_id")),
                "document_name": _meta_str(parent.get("document_name") or child.metadata.get("document_name")),
                "document_type": _meta_str(parent.get("document_type") or child.metadata.get("document_type")),
                "topic": _meta_str(parent.get("topic") or child.metadata.get("topic")),
                "authority_level": int(parent.get("authority_level") or child.metadata.get("authority_level") or 0),
                "source": _meta_str(parent.get("source") or child.metadata.get("source")),
                "file_path": _meta_str(parent.get("file_path") or child.metadata.get("file_path")),
                "page": int(child_page) if child_page else 0,
                "page_end": int(page_end) if page_end else 0,
                "parent_id": parent_id,
                "article_nos": ",".join(str(x) for x in article_nos),
                "article_title": title,
                "chapter": _meta_str(parent.get("chapter")),
                "section": _meta_str(parent.get("section")),
                "chunk_type": "parent",
                "merge_count": int(parent.get("merge_count") or 1),
                "hit_child_id": _meta_str(child.metadata.get("child_id")),
                "hit_article_no": int(child.metadata.get("article_no") or 0),
                "chunk_origin": _meta_str(parent.get("chunk_origin")),
            }
            _append(Document(page_content=content, metadata=metadata))
            continue
        # Fallback: keep child if parent missing.
        key = _meta_str((child.metadata or {}).get("child_id") or id(child))
        if key in seen_parent:
            continue
        seen_parent.add(key)
        _append(child)
    return output
