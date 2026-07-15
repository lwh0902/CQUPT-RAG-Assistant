"""Document page preview routes — renders PDF pages as PNG on demand."""

from __future__ import annotations

import logging
from pathlib import Path

import fitz
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from models import User
from security import get_current_user
from settings import POLICY_DOCUMENTS

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(get_current_user)],
)

CACHE_DIR = Path("./cache/pdf_pages")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def resolve_document_path(document_id: str) -> Path | None:
    """Look up a document's source file path by id from the policy whitelist."""
    for item in POLICY_DOCUMENTS:
        if item["document_id"] == document_id:
            path = Path(item["path"]).expanduser()
            return path if path.exists() else None
    return None


def _get_pdf_page_count(pdf_path: Path) -> int | None:
    try:
        doc = fitz.open(str(pdf_path))
        try:
            return doc.page_count
        finally:
            doc.close()
    except Exception:
        logging.exception("Failed to read page count for %s", pdf_path)
        return None


@router.get("")
def list_documents() -> dict[str, list[dict[str, object]]]:
    """Return metadata for every document in the policy whitelist."""
    items: list[dict[str, object]] = []
    for entry in POLICY_DOCUMENTS:
        path = Path(entry["path"]).expanduser()
        exists = path.exists()
        suffix = path.suffix.lower()
        is_pdf = suffix == ".pdf"
        page_count = _get_pdf_page_count(path) if (exists and is_pdf) else None
        items.append(
            {
                "document_id": entry["document_id"],
                "document_name": entry["document_name"],
                "document_type": entry.get("document_type", ""),
                "topic": entry.get("topic", ""),
                "authority_level": entry.get("authority_level", 0),
                "file_name": path.name,
                "file_size": path.stat().st_size if exists else 0,
                "file_exists": exists,
                "previewable": is_pdf and exists,
                "page_count": page_count,
            }
        )
    return {"documents": items}


def render_page_to_png(
    pdf_path: Path,
    page: int,
    *,
    thumbnail: bool = False,
) -> Path:
    """Render a single PDF page to a cached PNG file. Returns the cache path.

    Raises HTTPException(404) when the page number is out of range.
    """
    suffix = "_thumb" if thumbnail else ""
    cache_path = CACHE_DIR / f"{pdf_path.stem}_p{page:04d}{suffix}.png"
    if cache_path.exists():
        return cache_path

    document = fitz.open(str(pdf_path))
    try:
        if page < 1 or page > document.page_count:
            raise HTTPException(status_code=404, detail="页码超出文档范围")
        page_obj = document.load_page(page - 1)
        matrix = fitz.Matrix(0.4, 0.4) if thumbnail else fitz.Matrix(1.5, 1.5)
        pixmap = page_obj.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(str(cache_path))
    finally:
        document.close()

    return cache_path


@router.get("/{document_id}/page/{page}")
def get_document_page(document_id: str, page: int) -> FileResponse:
    if page < 1:
        raise HTTPException(status_code=422, detail="page 必须 >= 1")

    pdf_path = resolve_document_path(document_id)
    if pdf_path is None:
        raise HTTPException(status_code=404, detail="文档不存在或源文件已移除")
    if pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="该文档类型暂不支持预览")

    try:
        cache_path = render_page_to_png(pdf_path, page)
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Failed to render page %s of %s", page, document_id)
        raise HTTPException(status_code=500, detail=f"渲染失败：{exc}") from exc

    return FileResponse(cache_path, media_type="image/png")


@router.get("/{document_id}/page/{page}/thumbnail")
def get_document_page_thumbnail(document_id: str, page: int) -> FileResponse:
    if page < 1:
        raise HTTPException(status_code=422, detail="page 必须 >= 1")

    pdf_path = resolve_document_path(document_id)
    if pdf_path is None:
        raise HTTPException(status_code=404, detail="文档不存在或源文件已移除")
    if pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="该文档类型暂不支持预览")

    try:
        cache_path = render_page_to_png(pdf_path, page, thumbnail=True)
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Failed to render thumbnail %s of %s", page, document_id)
        raise HTTPException(status_code=500, detail=f"渲染失败：{exc}") from exc

    return FileResponse(cache_path, media_type="image/png")
