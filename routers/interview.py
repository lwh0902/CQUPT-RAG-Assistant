"""Interview assistant routes: generate / list / detail / delete question banks."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from database import engine
from models import InterviewQuestion, InterviewSession, User
from security import get_current_user
from services.interview import (
    clamp_text,
    extract_resume_text,
    generate_mcq_bank,
    generate_qa_bank,
    generate_targeted_mcq_bank,
    generate_weakness_report,
    search_interview_references,
    MAX_JD_CHARS,
    MAX_RESUME_CHARS,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/interview", tags=["interview"])

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def _serialize_session(session: InterviewSession, *, with_questions: bool) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": session.id,
        "company": session.company,
        "position": getattr(session, "position", "") or "",
        "jd_text": session.jd_text,
        "resume_filename": session.resume_filename,
        "reference_used": bool(getattr(session, "reference_used", False)),
        "references": json.loads(session.reference_json) if getattr(session, "reference_json", None) else [],
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


@router.post("/generate")
async def generate_interview_bank(
    company: str = Form(default=""),
    position: str = Form(default=""),
    jd_text: str = Form(...),
    resume_text: str = Form(default=""),
    resume_file: Optional[UploadFile] = File(default=None),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    jd_text = (jd_text or "").strip()
    if len(jd_text) < 10:
        raise HTTPException(status_code=400, detail="岗位 JD 至少需要 10 个字符")

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

    resolved_resume = clamp_text(resolved_resume, MAX_RESUME_CHARS)
    jd_text = clamp_text(jd_text, MAX_JD_CHARS)

    # 面经参考：联网搜索真实面试经验；无 key/无结果时自动纯模型生成。
    references_text, references = await search_interview_references(company, position)

    try:
        from settings import INTERVIEW_MCQ_REVIEW

        mcq_raw, qa = await asyncio.gather(
            asyncio.to_thread(
                generate_mcq_bank,
                company=company,
                position=position,
                jd_text=jd_text,
                resume_text=resolved_resume,
                references=references_text,
            ),
            asyncio.to_thread(
                generate_qa_bank,
                company=company,
                position=position,
                jd_text=jd_text,
                resume_text=resolved_resume,
                references=references_text,
            ),
        )
        # Second-pass review: drop/fix questions with wrong answer keys.
        from services.interview import review_mcq_bank

        mcq = await asyncio.to_thread(review_mcq_bank, mcq_raw, enabled=INTERVIEW_MCQ_REVIEW)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    with Session(engine) as db:
        session = InterviewSession(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            company=(company or "").strip()[:100],
            position=(position or "").strip()[:100],
            jd_text=jd_text,
            resume_text=resolved_resume,
            resume_filename=resume_filename,
            reference_used=bool(references),
            reference_json=json.dumps(references, ensure_ascii=False) if references else None,
        )
        db.add(session)
        for ordinal, item in enumerate(mcq, start=1):
            db.add(
                InterviewQuestion(
                    session_id=session.id,
                    qtype="mcq",
                    ordinal=ordinal,
                    round=1,
                    payload_json=json.dumps(item, ensure_ascii=False),
                )
            )
        for ordinal, item in enumerate(qa, start=1):
            db.add(
                InterviewQuestion(
                    session_id=session.id,
                    qtype="qa",
                    ordinal=ordinal,
                    round=1,
                    payload_json=json.dumps(item, ensure_ascii=False),
                )
            )
        db.commit()
        db.refresh(session)
        return _serialize_session(session, with_questions=True)


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


def _load_round1_mcq(session: InterviewSession, wrong_indices: list[int]) -> list[dict[str, Any]]:
    """Resolve wrong MCQ items (round 1) by 1-based indices; empty list = all wrong handled by caller."""
    round1 = [q for q in sorted(session.questions, key=lambda q: (q.round, q.ordinal)) if q.qtype == "mcq" and q.round == 1]
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

            items_raw = await asyncio.to_thread(
                generate_targeted_mcq_bank,
                company=session.company,
                position=getattr(session, "position", "") or "",
                jd_text=session.jd_text,
                resume_text=session.resume_text,
                weak_points=weak_points_source,
                count=10,
            )
            from services.interview import review_mcq_bank

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
