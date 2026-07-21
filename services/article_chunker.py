"""Clause-level (article) parsing and parent/child preview helpers.

This module is intentionally independent of Chroma rebuild. It only turns
page-level policy text into structured articles for human review.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

# 第X章 / 第X节 / 第X条，兼容全角数字与空格噪声。
_CHAPTER_RE = re.compile(
    r"第\s*([零〇一二三四五六七八九十百千0-9]+)\s*章\s*([^\n第]{0,40})"
)
_SECTION_RE = re.compile(
    r"第\s*([零〇一二三四五六七八九十百千0-9]+)\s*节\s*([^\n第]{0,40})"
)
_ARTICLE_RE = re.compile(
    r"(?:(?<=\n)|(?<=^)|(?<=\s))"
    r"第\s*([零〇一二三四五六七八九十百千0-9]+)\s*条"
    r"([^\n]{0,80})"
)

_CN_NUM = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def chinese_numeral_to_int(raw: str) -> int | None:
    text = (raw or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    # Simple 1–99 style numerals common in policy docs.
    if text == "十":
        return 10
    if text.startswith("十"):
        ones = _CN_NUM.get(text[1:], 0) if len(text) > 1 else 0
        return 10 + ones
    if "十" in text:
        left, _, right = text.partition("十")
        tens = _CN_NUM.get(left, 0)
        ones = _CN_NUM.get(right, 0) if right else 0
        return tens * 10 + ones
    if text in _CN_NUM:
        return _CN_NUM[text]
    return None


def _clean_heading(text: str) -> str:
    value = re.sub(r"\s+", " ", (text or "").strip())
    value = value.strip(" ：:.-—_")
    return value[:60]


def _looks_like_article_title(text: str) -> bool:
    """Filter out body sentences mistakenly captured as titles."""
    value = _clean_heading(text)
    if not value or len(value) > 36:
        return False
    if value.startswith("（") or value.startswith("("):
        return False
    # Body sentences / truncated body lines.
    if value.endswith("。") or value.endswith("；") or value.endswith(";"):
        return False
    if value.endswith(("的", "可", "和", "与", "及", "学", "，", ",")):
        return False
    title_cues = ("条件", "程序", "范围", "标准", "原则", "对象", "细则", "类型")
    if any(cue in value for cue in title_cues):
        return True
    if "\u201c" in value or "\u201d" in value:
        return True
    # Short heading-like fragment without clause verbs.
    bodyish = ("用于", "应当", "可以", "不得", "适用", "根据", "为了", "本办法", "本规定")
    if any(token in value for token in bodyish):
        return False
    return len(value) <= 20


def _normalize_page_text(text: str) -> str:
    # Keep newlines for heading detection; collapse repeated spaces.
    lines = []
    for line in (text or "").splitlines():
        cleaned = re.sub(r"[ \t]+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def parse_articles_from_pages(
    pages: Iterable[dict[str, Any]],
    *,
    document_id: str,
    document_name: str,
) -> list[dict[str, Any]]:
    """Parse page dicts `{page, text}` into article-level parent chunks."""
    # Build a continuous corpus with page markers so cross-page articles work.
    parts: list[str] = []
    page_spans: list[tuple[int, int, int]] = []  # start, end, page
    cursor = 0
    for item in pages:
        page_no = int(item["page"])
        text = _normalize_page_text(str(item.get("text") or ""))
        if not text:
            continue
        if parts:
            parts.append("\n")
            cursor += 1
        start = cursor
        parts.append(text)
        cursor += len(text)
        page_spans.append((start, cursor, page_no))

    corpus = "".join(parts)
    if not corpus.strip():
        return []

    chapter_marks = [
        (match.start(), chinese_numeral_to_int(match.group(1)), _clean_heading(match.group(2)))
        for match in _CHAPTER_RE.finditer(corpus)
    ]
    section_marks = [
        (match.start(), chinese_numeral_to_int(match.group(1)), _clean_heading(match.group(2)))
        for match in _SECTION_RE.finditer(corpus)
    ]
    article_matches = list(_ARTICLE_RE.finditer(corpus))
    if not article_matches:
        return []

    def lookup_heading(marks: list[tuple[int, int | None, str]], pos: int) -> tuple[int | None, str]:
        current_no, current_title = None, ""
        for start, number, title in marks:
            if start <= pos:
                current_no, current_title = number, title
            else:
                break
        return current_no, current_title

    def page_for_pos(pos: int) -> int | None:
        for start, end, page in page_spans:
            if start <= pos < end:
                return page
        if page_spans and pos >= page_spans[-1][1]:
            return page_spans[-1][2]
        return None

    articles: list[dict[str, Any]] = []
    for index, match in enumerate(article_matches):
        art_no = chinese_numeral_to_int(match.group(1))
        inline_title = _clean_heading(match.group(2))
        title = inline_title if _looks_like_article_title(inline_title) else ""
        body_start = match.start()
        body_end = article_matches[index + 1].start() if index + 1 < len(article_matches) else len(corpus)
        raw = corpus[body_start:body_end].strip()
        if not raw:
            continue
        # Many handbook PDFs put the article title on the next line after "第X条".
        if not title:
            after = corpus[match.end():body_end].lstrip(" \t\r\n：:")
            first_line = after.splitlines()[0].strip() if after else ""
            if _looks_like_article_title(first_line):
                title = _clean_heading(first_line)
        chapter_no, chapter_title = lookup_heading(chapter_marks, body_start)
        section_no, section_title = lookup_heading(section_marks, body_start)
        page_start = page_for_pos(body_start)
        page_end = page_for_pos(max(body_start, body_end - 1))
        article_id = (
            f"{document_id}:art:{art_no}:{page_start}"
            if art_no is not None
            else f"{document_id}:span:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]}"
        )
        articles.append(
            {
                "article_id": article_id,
                "document_id": document_id,
                "document_name": document_name,
                "article_no": art_no,
                "article_title": title,
                "chapter_no": chapter_no,
                "chapter": chapter_title,
                "section_no": section_no,
                "section": section_title,
                "page_start": page_start,
                "page_end": page_end,
                "char_count": len(raw),
                "text": raw,
            }
        )
    return articles


def split_parent_into_children(
    parent: dict[str, Any],
    *,
    max_chars: int = 180,
) -> list[dict[str, Any]]:
    """Split one article parent into smaller child retrieval units."""
    text = (parent.get("text") or "").strip()
    parent_id = parent.get("article_id") or "unknown"
    if not text:
        return []

    # Prefer itemized clauses; fallback to punctuation windows.
    pieces = re.split(r"(?=(?:（[一二三四五六七八九十0-9]+）|\([0-9]+\)))", text)
    pieces = [p.strip() for p in pieces if p and p.strip()]
    if len(pieces) <= 1:
        pieces = re.split(r"(?<=[。；;])", text)
        pieces = [p.strip() for p in pieces if p and p.strip()]

    children: list[dict[str, Any]] = []
    buf = ""
    for piece in pieces:
        candidate = f"{buf}{piece}".strip() if buf else piece
        if len(candidate) <= max_chars or not buf:
            buf = candidate
            if len(buf) >= max_chars:
                children.append(buf)
                buf = ""
        else:
            children.append(buf)
            buf = piece
    if buf:
        children.append(buf)

    output: list[dict[str, Any]] = []
    for idx, child_text in enumerate(children, start=1):
        output.append(
            {
                "child_id": f"{parent_id}:c{idx}",
                "parent_id": parent_id,
                "document_id": parent.get("document_id"),
                "document_name": parent.get("document_name"),
                "article_no": parent.get("article_no"),
                "article_title": parent.get("article_title"),
                "page_start": parent.get("page_start"),
                "page_end": parent.get("page_end"),
                "child_index": idx,
                "char_count": len(child_text),
                "text": child_text,
            }
        )
    return output


def summarize_articles(articles: list[dict[str, Any]]) -> dict[str, Any]:
    if not articles:
        return {
            "article_count": 0,
            "with_title_rate": 0.0,
            "avg_chars": 0.0,
            "median_chars": 0.0,
            "page_span_gt1": 0,
        }
    titled = sum(1 for item in articles if item.get("article_title"))
    spans = sum(1 for item in articles if (item.get("page_end") or 0) != (item.get("page_start") or 0))
    lengths = sorted(int(item.get("char_count") or 0) for item in articles)
    avg = sum(lengths) / len(lengths)
    median = lengths[len(lengths) // 2]
    return {
        "article_count": len(articles),
        "with_title_rate": titled / len(articles),
        "avg_chars": round(avg, 1),
        "median_chars": median,
        "page_span_gt1": spans,
    }


def _same_merge_group(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Only merge consecutive articles in the same document chapter/section."""
    if left.get("document_id") != right.get("document_id"):
        return False
    if (left.get("chapter_no"), left.get("chapter")) != (
        right.get("chapter_no"),
        right.get("chapter"),
    ):
        return False
    if (left.get("section_no"), left.get("section")) != (
        right.get("section_no"),
        right.get("section"),
    ):
        return False
    # Keep article numbers roughly contiguous when available.
    left_no = left.get("article_no")
    right_no = right.get("article_no")
    if isinstance(left_no, int) and isinstance(right_no, int):
        if right_no < left_no or right_no - left_no > 3:
            return False
    # Avoid jumping too far across pages inside one parent.
    left_page = left.get("page_end") or left.get("page_start") or 0
    right_page = right.get("page_start") or right.get("page_end") or 0
    try:
        if int(right_page) - int(left_page) > 2:
            return False
    except (TypeError, ValueError):
        pass
    return True


def _build_merged_parent(group: list[dict[str, Any]], *, group_index: int) -> dict[str, Any]:
    first, last = group[0], group[-1]
    texts = [str(item.get("text") or "").strip() for item in group if str(item.get("text") or "").strip()]
    merged_text = "\n\n".join(texts)
    titles = [str(item.get("article_title") or "").strip() for item in group if item.get("article_title")]
    article_nos = [item.get("article_no") for item in group if item.get("article_no") is not None]
    doc_id = first.get("document_id") or "doc"
    parent_id = f"{doc_id}:parent:{group_index}:{first.get('page_start')}-{last.get('page_end')}"
    if len(group) == 1:
        label = titles[0] if titles else f"第{first.get('article_no')}条"
    else:
        no_label = ""
        if article_nos:
            no_label = f"第{article_nos[0]}–{article_nos[-1]}条"
        topic = first.get("section") or first.get("chapter") or "相关条款"
        label = f"{topic} {no_label}".strip()
    return {
        "parent_id": parent_id,
        "document_id": first.get("document_id"),
        "document_name": first.get("document_name"),
        "chapter_no": first.get("chapter_no"),
        "chapter": first.get("chapter") or "",
        "section_no": first.get("section_no"),
        "section": first.get("section") or "",
        "article_nos": article_nos,
        "article_titles": titles,
        "title": label,
        "page_start": first.get("page_start"),
        "page_end": last.get("page_end"),
        "merge_count": len(group),
        "source_article_ids": [item.get("article_id") for item in group],
        "char_count": len(merged_text),
        "text": merged_text,
    }


def merge_articles_into_parents(
    articles: list[dict[str, Any]],
    *,
    min_parent_chars: int = 220,
    max_parent_chars: int = 900,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge short consecutive articles into larger parents; keep articles as children.

    Policy:
    - Long article (>= min_parent_chars): stays a solo parent.
    - Short articles: greedily merge with following same-group shorts until
      reaching min_parent_chars, without exceeding max_parent_chars.
    - Children are always the original article units for retrieval.
    """
    if not articles:
        return [], []

    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    groups: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(articles):
        current = articles[index]
        current_len = int(current.get("char_count") or len(current.get("text") or ""))

        # Sufficiently large article: keep as its own parent.
        if current_len >= min_parent_chars:
            group = [current]
            index += 1
        else:
            group = [current]
            index += 1
            running = current_len
            while index < len(articles):
                nxt = articles[index]
                nxt_len = int(nxt.get("char_count") or len(nxt.get("text") or ""))
                # Do not swallow a long article into a short group.
                if nxt_len >= min_parent_chars:
                    break
                if not _same_merge_group(group[-1], nxt):
                    break
                if running + nxt_len + 2 > max_parent_chars:
                    break
                group.append(nxt)
                running += nxt_len + 2
                index += 1
                if running >= min_parent_chars:
                    break

        groups.append(group)

    # Second pass: attach leftover tiny groups to neighboring same-group parents.
    compact_groups: list[list[dict[str, Any]]] = []
    pending_tiny: list[dict[str, Any]] | None = None

    def group_chars(items: list[dict[str, Any]]) -> int:
        return sum(int(item.get("char_count") or len(item.get("text") or "")) for item in items)

    for group in groups:
        # Flush a pending tiny group into the current group when safe.
        if pending_tiny is not None:
            if (
                _same_merge_group(pending_tiny[-1], group[0])
                and group_chars(pending_tiny) + group_chars(group) + 2 <= max_parent_chars
            ):
                group = [*pending_tiny, *group]
                pending_tiny = None
            else:
                compact_groups.append(pending_tiny)
                pending_tiny = None

        group_len = group_chars(group)
        if group_len < min_parent_chars:
            # Prefer merge backward into previous parent.
            if (
                compact_groups
                and _same_merge_group(compact_groups[-1][-1], group[0])
                and group_chars(compact_groups[-1]) + group_len + 2 <= max_parent_chars
            ):
                compact_groups[-1].extend(group)
            else:
                # Otherwise hold and try merge forward into the next group.
                pending_tiny = group
            continue
        compact_groups.append(group)

    if pending_tiny is not None:
        if (
            compact_groups
            and _same_merge_group(compact_groups[-1][-1], pending_tiny[0])
            and group_chars(compact_groups[-1]) + group_chars(pending_tiny) + 2 <= max_parent_chars
        ):
            compact_groups[-1].extend(pending_tiny)
        else:
            compact_groups.append(pending_tiny)

    for group_index, group in enumerate(compact_groups, start=1):
        parent = _build_merged_parent(group, group_index=group_index)
        parents.append(parent)
        for child_index, article in enumerate(group, start=1):
            child_text = str(article.get("text") or "").strip()
            children.append(
                {
                    "child_id": f"{parent['parent_id']}:art:{article.get('article_no') or child_index}",
                    "parent_id": parent["parent_id"],
                    "source_article_id": article.get("article_id"),
                    "document_id": article.get("document_id"),
                    "document_name": article.get("document_name"),
                    "article_no": article.get("article_no"),
                    "article_title": article.get("article_title") or "",
                    "chapter": article.get("chapter") or "",
                    "section": article.get("section") or "",
                    "page_start": article.get("page_start"),
                    "page_end": article.get("page_end"),
                    "child_index": child_index,
                    "char_count": len(child_text),
                    "text": child_text,
                }
            )

    return parents, children


def summarize_length_distribution(items: list[dict[str, Any]], field: str = "char_count") -> dict[str, Any]:
    lengths = sorted(int(item.get(field) or 0) for item in items)
    if not lengths:
        return {"n": 0}
    def pct(p: float) -> int:
        return lengths[min(len(lengths) - 1, int(len(lengths) * p))]
    bins = {
        "lt_100": sum(1 for x in lengths if x < 100),
        "100_199": sum(1 for x in lengths if 100 <= x < 200),
        "200_399": sum(1 for x in lengths if 200 <= x < 400),
        "400_799": sum(1 for x in lengths if 400 <= x < 800),
        "ge_800": sum(1 for x in lengths if x >= 800),
    }
    return {
        "n": len(lengths),
        "avg": round(sum(lengths) / len(lengths), 1),
        "median": pct(0.5),
        "p75": pct(0.75),
        "p90": pct(0.9),
        "min": lengths[0],
        "max": lengths[-1],
        "bins": bins,
    }
