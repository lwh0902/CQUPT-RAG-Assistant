"""Interview assistant routes: generate / list / detail / delete question banks."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
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
        "jd_text": session.jd_text,
        "resume_filename": session.resume_filename,
        "created_at": session.created_at.isoformat() if session.created_at else None,
    }
    if with_questions:
        data["resume_text"] = session.resume_text
        data["mcq"] = [
            json.loads(q.payload_json)
            for q in sorted(session.questions, key=lambda q: q.ordinal)
            if q.qtype == "mcq"
        ]
        data["qa"] = [
            json.loads(q.payload_json)
            for q in sorted(session.questions, key=lambda q: q.ordinal)
            if q.qtype == "qa"
        ]
    else:
        data["mcq_count"] = sum(1 for q in session.questions if q.qtype == "mcq")
        data["qa_count"] = sum(1 for q in session.questions if q.qtype == "qa")
    return data


@router.post("/generate")
async def generate_interview_bank(
    company: str = Form(default=""),
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
        raise HTTPException(status_code=400, detail="请上传简历 PDF 或粘贴简历文本（至少 20 个字符）")

    resolved_resume = clamp_text(resolved_resume, MAX_RESUME_CHARS)
    jd_text = clamp_text(jd_text, MAX_JD_CHARS)

    try:
        mcq, qa = await asyncio.gather(
            asyncio.to_thread(
                generate_mcq_bank,
                company=company,
                jd_text=jd_text,
                resume_text=resolved_resume,
            ),
            asyncio.to_thread(
                generate_qa_bank,
                company=company,
                jd_text=jd_text,
                resume_text=resolved_resume,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    with Session(engine) as db:
        session = InterviewSession(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            company=(company or "").strip()[:100],
            jd_text=jd_text,
            resume_text=resolved_resume,
            resume_filename=resume_filename,
        )
        db.add(session)
        for ordinal, item in enumerate(mcq, start=1):
            db.add(
                InterviewQuestion(
                    session_id=session.id,
                    qtype="mcq",
                    ordinal=ordinal,
                    payload_json=json.dumps(item, ensure_ascii=False),
                )
            )
        for ordinal, item in enumerate(qa, start=1):
            db.add(
                InterviewQuestion(
                    session_id=session.id,
                    qtype="qa",
                    ordinal=ordinal,
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
