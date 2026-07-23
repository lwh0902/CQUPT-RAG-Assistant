"""Interview assistant routes: generate (SSE) / list / detail / delete / tutor."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from database import engine
from models import InterviewQuestion, InterviewSession, User
from security import get_current_user
from services.interview import (
    MAX_JD_CHARS,
    MAX_RESUME_CHARS,
    build_interview_bank,
    clamp_text,
    extract_resume_text,
    generate_targeted_mcq_bank,
    generate_weakness_report,
    search_interview_references,
    stream_interview_tutor,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/interview", tags=["interview"])

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def _parse_reference_blob(raw: Optional[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Support both legacy list refs and v3 {sources, real_questions} blob."""
    if not raw:
        return [], []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], []
    if isinstance(data, list):
        return data, []
    if isinstance(data, dict):
        sources = data.get("sources") or []
        real_questions = data.get("real_questions") or []
        if not isinstance(sources, list):
            sources = []
        if not isinstance(real_questions, list):
            real_questions = []
        return sources, real_questions
    return [], []


def _serialize_session(session: InterviewSession, *, with_questions: bool) -> dict[str, Any]:
    sources, real_questions = _parse_reference_blob(getattr(session, "reference_json", None))
    data: dict[str, Any] = {
        "id": session.id,
        "company": session.company,
        "position": getattr(session, "position", "") or "",
        "jd_text": session.jd_text,
        "resume_filename": session.resume_filename,
        "reference_used": bool(getattr(session, "reference_used", False)),
        "references": sources,
        "real_questions": real_questions,
        "report_text": getattr(session, "report_text", None),
        "created_at": session.created_at.isoformat() if session.created_at else None,
    }
    if with_questions:
        data["resume_text"] = session.resume_text
        ordered = sorted(session.questions, key=lambda q: (q.round, q.ordinal))
        data["mcq"] = [
            {**json.loads(q.payload_json), "round": q.round}
            for q in ordered
            if q.qtype == "mcq"
        ]
        data["qa"] = [
            json.loads(q.payload_json)
            for q in ordered
            if q.qtype == "qa"
        ]
    else:
        data["mcq_count"] = sum(1 for q in session.questions if q.qtype == "mcq")
        data["qa_count"] = sum(1 for q in session.questions if q.qtype == "qa")
    return data


def _sse(event: str, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n"


async def _resolve_resume(
    resume_text: str,
    resume_file: Optional[UploadFile],
) -> tuple[str, Optional[str]]:
    resolved_resume = (resume_text or "").strip()
    resume_filename: Optional[str] = None
    if resume_file is not None and resume_file.filename:
        content = await resume_file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="简历文件不能超过 5MB")
        try:
            resolved_resume = await asyncio.to_thread(
                extract_resume_text, content, resume_file.filename
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        resume_filename = resume_file.filename

    if len(resolved_resume) < 20:
        raise HTTPException(status_code=400, detail="请上传简历 PDF/DOCX 或粘贴简历文本（至少 20 个字符）")
    return clamp_text(resolved_resume, MAX_RESUME_CHARS), resume_filename


@router.post("/generate")
async def generate_interview_bank(
    company: str = Form(default=""),
    position: str = Form(default=""),
    jd_text: str = Form(...),
    resume_text: str = Form(default=""),
    resume_file: Optional[UploadFile] = File(default=None),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Legacy non-stream generate (kept for compatibility / simple clients)."""
    jd_text = clamp_text((jd_text or "").strip(), MAX_JD_CHARS)
    if len(jd_text) < 10:
        raise HTTPException(status_code=400, detail="岗位 JD 至少需要 10 个字符")

    resolved_resume, resume_filename = await _resolve_resume(resume_text, resume_file)
    references_text, references = await search_interview_references(company, position)

    try:
        result = await asyncio.to_thread(
            build_interview_bank,
            company=company,
            position=position,
            jd_text=jd_text,
            resume_text=resolved_resume,
            resume_filename=resume_filename,
            user_id=current_user.id,
            references_text=references_text,
            references=references,
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return result


@router.post("/generate/stream")
async def generate_interview_bank_stream(
    company: str = Form(default=""),
    position: str = Form(default=""),
    jd_text: str = Form(...),
    resume_text: str = Form(default=""),
    resume_file: Optional[UploadFile] = File(default=None),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """SSE generate with real stage events."""
    jd_text = clamp_text((jd_text or "").strip(), MAX_JD_CHARS)
    if len(jd_text) < 10:
        raise HTTPException(status_code=400, detail="岗位 JD 至少需要 10 个字符")

    resolved_resume, resume_filename = await _resolve_resume(resume_text, resume_file)
    user_id = current_user.id

    async def event_stream() -> AsyncIterator[str]:
        yield _sse("stage", {"stage": "parse", "message": "正在解析简历与岗位信息…", "progress": 4})

        yield _sse("stage", {"stage": "search", "message": "正在联网搜索真实面经…", "progress": 12})
        try:
            references_text, references = await search_interview_references(company, position)
        except Exception:
            logger.warning("stream search failed", exc_info=True)
            references_text, references = "", []

        if references:
            yield _sse(
                "stage",
                {
                    "stage": "search",
                    "message": f"已找到 {len(references)} 条面经来源，开始提炼真题…",
                    "progress": 18,
                    "ref_count": len(references),
                },
            )
        else:
            yield _sse(
                "stage",
                {
                    "stage": "search",
                    "message": "未启用联网或未搜到面经，将纯模型出题…",
                    "progress": 18,
                    "ref_count": 0,
                },
            )

        loop = asyncio.get_running_loop()
        stage_q: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()

        def on_stage(key: str, message: str, progress: int, extra: dict[str, Any]) -> None:
            payload = {"stage": key, "message": message, "progress": progress, **(extra or {})}
            stage_q.put(("stage", payload))

        def worker() -> None:
            try:
                result = build_interview_bank(
                    company=company,
                    position=position,
                    jd_text=jd_text,
                    resume_text=resolved_resume,
                    resume_filename=resume_filename,
                    user_id=user_id,
                    references_text=references_text,
                    references=references,
                    on_stage=on_stage,
                )
                stage_q.put(("done", result))
            except Exception as exc:  # noqa: BLE001
                logger.exception("interview stream generate failed")
                stage_q.put(("error", {"detail": str(exc) or "题库生成失败"}))
            finally:
                stage_q.put(None)

        await loop.run_in_executor(None, lambda: None)  # warm default executor
        fut = loop.run_in_executor(None, worker)

        while True:
            item = await asyncio.to_thread(stage_q.get)
            if item is None:
                break
            event, payload = item
            if event == "stage":
                # Interpolate QA chunk progress 58→88
                if payload.get("stage") == "qa" and payload.get("total_chunks"):
                    chunk = int(payload.get("chunk") or 1)
                    total = max(1, int(payload.get("total_chunks") or 1))
                    payload["progress"] = 58 + int(30 * chunk / total)
                yield _sse("stage", payload)
            elif event == "done":
                yield _sse("done", payload)
            elif event == "error":
                yield _sse("error", payload)

        # Surface unexpected worker crash if queue ended without done/error
        try:
            await fut
        except Exception:
            logger.exception("interview generate worker crashed")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions")
def list_interview_sessions(
    current_user: User = Depends(get_current_user),
) -> dict[str, list[dict[str, Any]]]:
    with Session(engine) as db:
        sessions = db.scalars(
            select(InterviewSession)
            .where(InterviewSession.user_id == current_user.id)
            .order_by(desc(InterviewSession.created_at))
            .limit(30)
        ).all()
        return {"sessions": [_serialize_session(s, with_questions=False) for s in sessions]}


@router.get("/sessions/{session_id}")
def get_interview_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    with Session(engine) as db:
        session = db.get(InterviewSession, session_id)
        if session is None or session.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="题库不存在")
        return _serialize_session(session, with_questions=True)


@router.delete("/sessions/{session_id}")
def delete_interview_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    with Session(engine) as db:
        session = db.get(InterviewSession, session_id)
        if session is None or session.user_id != current_user.id:
            return {"id": session_id, "status": "not_found"}
        db.delete(session)
        db.commit()
        return {"id": session_id, "status": "deleted"}


class WrongAnswersPayload(BaseModel):
    wrong_indices: list[int] = Field(default_factory=list, max_length=60)


class TutorMessage(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=4000)


class TutorPayload(BaseModel):
    messages: list[TutorMessage] = Field(min_length=1, max_length=20)


def _load_round1_mcq(session: InterviewSession, wrong_indices: list[int]) -> list[dict[str, Any]]:
    """Resolve wrong MCQ items (round 1) by 1-based indices."""
    round1 = [
        q
        for q in sorted(session.questions, key=lambda q: (q.round, q.ordinal))
        if q.qtype == "mcq" and q.round == 1
    ]
    items: list[dict[str, Any]] = []
    for index in wrong_indices:
        if 1 <= index <= len(round1):
            items.append(json.loads(round1[index - 1].payload_json))
    return items


@router.post("/sessions/{session_id}/report")
async def create_weakness_report(
    session_id: str,
    payload: WrongAnswersPayload,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    with Session(engine) as db:
        session = db.get(InterviewSession, session_id)
        if session is None or session.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="题库不存在")
        wrong_items = _load_round1_mcq(session, payload.wrong_indices)
        if not wrong_items:
            raise HTTPException(status_code=400, detail="没有可分析的错题")

        try:
            report = await asyncio.to_thread(generate_weakness_report, wrong_items)
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if not report:
            raise HTTPException(status_code=502, detail="报告生成失败，请重试")

        session.report_text = report
        db.add(session)
        db.commit()
        return {"id": session_id, "report": report}


@router.post("/sessions/{session_id}/regenerate-mcq")
async def regenerate_targeted_mcq(
    session_id: str,
    payload: WrongAnswersPayload,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    with Session(engine) as db:
        session = db.get(InterviewSession, session_id)
        if session is None or session.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="题库不存在")

        wrong_items = _load_round1_mcq(session, payload.wrong_indices)
        weak_points_source = ""
        if session.report_text:
            weak_points_source = session.report_text
        elif wrong_items:
            weak_points_source = "；".join(item.get("question", "") for item in wrong_items)
        if not weak_points_source:
            raise HTTPException(status_code=400, detail="请先提交答卷或生成薄弱点报告")

        try:
            from settings import INTERVIEW_MCQ_REVIEW
            from services.interview import review_mcq_bank

            items_raw = await asyncio.to_thread(
                generate_targeted_mcq_bank,
                company=session.company,
                position=getattr(session, "position", "") or "",
                jd_text=session.jd_text,
                resume_text=session.resume_text,
                weak_points=weak_points_source,
                count=10,
            )
            items = await asyncio.to_thread(review_mcq_bank, items_raw, enabled=INTERVIEW_MCQ_REVIEW)
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        next_round = max((q.round for q in session.questions), default=1) + 1
        for ordinal, item in enumerate(items, start=1):
            db.add(
                InterviewQuestion(
                    session_id=session.id,
                    qtype="mcq",
                    ordinal=ordinal,
                    round=next_round,
                    payload_json=json.dumps(item, ensure_ascii=False),
                )
            )
        db.commit()
        db.refresh(session)
        return {
            "id": session_id,
            "round": next_round,
            "mcq": items,
        }


@router.post("/tutor/stream")
async def interview_tutor_stream(
    payload: TutorPayload,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Lightweight knowledge tutor for the interview page (no auto question context)."""
    _ = current_user
    messages = [{"role": m.role, "content": m.content} for m in payload.messages]

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event_type, content in stream_interview_tutor(messages):
                if event_type == "token":
                    yield _sse("token", {"content": content})
                elif event_type == "error":
                    yield _sse("error", {"detail": content or "讲解失败"})
                elif event_type == "done":
                    yield _sse("done", {})
        except ValueError as exc:
            yield _sse("error", {"detail": str(exc)})
        except Exception:
            logger.exception("interview tutor failed")
            yield _sse("error", {"detail": "讲解失败，请重试"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
