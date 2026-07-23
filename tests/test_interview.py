"""Interview assistant: extraction + generation validation (LLM mocked)."""

from __future__ import annotations

import json

import pytest

from services import interview
from services.interview import (
    clamp_text,
    extract_resume_text,
    generate_mcq_bank,
    generate_qa_bank,
)


def test_extract_resume_text_from_plain_text() -> None:
    text = extract_resume_text("张三\n计算机专业\n熟悉 Python、MySQL".encode("utf-8"), "resume.txt")
    assert "Python" in text


def test_extract_resume_text_rejects_broken_pdf() -> None:
    with pytest.raises(ValueError):
        extract_resume_text(b"not-a-pdf", "resume.pdf")


def test_clamp_text_truncates() -> None:
    text = clamp_text("字" * 100, 10)
    assert len(text) == 10
    assert text.endswith("…")


def _valid_mcq_payload(count: int = 20) -> str:
    return json.dumps(
        {
            "questions": [
                {
                    "question": f"Python GIL 的作用是什么？（{i}）",
                    "options": {"A": "加速多线程", "B": "限制同一时刻只有一个线程执行字节码", "C": "内存管理", "D": "垃圾回收"},
                    "answer": "B",
                    "analysis": "GIL 保证同一时刻只有一个线程执行 Python 字节码，因此多线程对 CPU 密集任务无效。",
                }
                for i in range(count)
            ]
        },
        ensure_ascii=False,
    )


def _valid_qa_payload(count: int = 20) -> str:
    return json.dumps(
        {
            "questions": [
                {
                    "question": f"讲讲你项目里的检索架构？（{i}）",
                    "spoken_answer": "我们的检索是向量加 BM25 混合召回，再按关键词精排，最后过证据门决定是否回答。",
                    "analysis": "考察系统设计能力与对口岗位匹配度：召回、排序、门控三层是否讲得清。",
                }
                for i in range(count)
            ]
        },
        ensure_ascii=False,
    )


def test_generate_mcq_bank_parses_valid_json(monkeypatch) -> None:
    monkeypatch.setattr(interview, "_call_llm_json", lambda prompt: _valid_mcq_payload())
    items = generate_mcq_bank(company="字节", jd_text="后端开发", resume_text="熟悉 Python")
    assert len(items) == 20
    assert items[0]["answer"] in {"A", "B", "C", "D"}
    assert "analysis" in items[0]


def test_generate_qa_bank_parses_valid_json(monkeypatch) -> None:
    monkeypatch.setattr(interview, "_call_llm_json", lambda prompt: _valid_qa_payload(count=15))
    items = generate_qa_bank(company="", jd_text="后端开发岗位 JD 内容", resume_text="熟悉检索系统", count=15)
    assert len(items) == 15
    assert items[0]["spoken_answer"]


def test_generate_bank_retries_then_fails(monkeypatch) -> None:
    calls = {"n": 0}

    def bad(prompt):
        calls["n"] += 1
        return "这不是 JSON"

    monkeypatch.setattr(interview, "_call_llm_json", bad)
    with pytest.raises(ValueError):
        generate_mcq_bank(company="", jd_text="x", resume_text="y")
    assert calls["n"] == 2


def test_generate_bank_tolerates_json_fence(monkeypatch) -> None:
    monkeypatch.setattr(
        interview, "_call_llm_json", lambda prompt: "```json\n" + _valid_mcq_payload() + "\n```"
    )
    items = generate_mcq_bank(company="", jd_text="x", resume_text="y")
    assert len(items) == 20


def test_extract_resume_text_from_docx() -> None:
    import io
    import zipfile

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body>'
        '<w:p><w:r><w:t>罗炜皓</w:t></w:r></w:p>'
        '<w:p><w:r><w:t>重庆邮电大学 计算机科学与技术</w:t></w:r></w:p>'
        '</w:body></w:document>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    text = extract_resume_text(buffer.getvalue(), "resume.docx")
    assert "罗炜皓" in text
    assert "重庆邮电大学" in text


def test_extract_resume_text_rejects_broken_docx() -> None:
    import pytest

    with pytest.raises(ValueError):
        extract_resume_text(b"not-a-zip", "resume.docx")


def test_qa_item_accepts_category() -> None:
    from services.interview import QaItem

    item = QaItem(
        question="讲讲你的项目",
        category="项目深挖",
        spoken_answer="我们的检索是向量加 BM25 混合召回，再按关键词精排。",
        analysis="考察系统设计能力。",
    )
    assert item.category == "项目深挖"


def test_generate_weakness_report_returns_markdown(monkeypatch) -> None:
    monkeypatch.setattr(interview, "_call_llm_json", lambda prompt: "## 薄弱知识点\n- Python GIL 理解不深")
    report = interview.generate_weakness_report(
        [{"question": "GIL 的作用？", "options": {"A": "x"}, "answer": "B"}]
    )
    assert "薄弱" in report


def test_generate_weakness_report_empty() -> None:
    assert interview.generate_weakness_report([]) == ""


def test_generate_targeted_mcq_uses_weak_points(monkeypatch) -> None:
    captured = {}

    def fake(prompt):
        captured["prompt"] = prompt
        return _valid_mcq_payload()

    monkeypatch.setattr(interview, "_call_llm_json", fake)
    items = interview.generate_targeted_mcq_bank(
        company="", position="", jd_text="后端", resume_text="Python",
        weak_points="GIL、数据库索引",
    )
    assert len(items) == 10
    assert "GIL、数据库索引" in captured["prompt"]


def test_reference_search_fallback_without_provider() -> None:
    import asyncio

    async def run():
        return await interview.search_interview_references("", "")

    text, refs = asyncio.run(run())
    assert text == "" and refs == []


def test_references_block_format() -> None:
    block = interview._references_block("面经内容")
    assert "面经参考" in block
    assert interview._references_block("") == ""


def test_suspicious_options_detect_duplicates() -> None:
    dup = {
        "A": "await session.execute(select(User)).scalars().all()",
        "B": "session.execute(await select(User)).scalars().all()",
        "C": "session.execute(select(User)).scalars().all()",
        "D": "await session.execute(select(User)).scalars().all()",
    }
    assert interview._options_are_suspicious(dup) is True

    ok = {"A": "加速多线程", "B": "限制单线程执行字节码", "C": "内存管理", "D": "垃圾回收"}
    assert interview._options_are_suspicious(ok) is False


def test_generate_bank_drops_duplicate_option_items(monkeypatch) -> None:
    payload = json.loads(_valid_mcq_payload())
    payload["questions"][0]["options"]["D"] = payload["questions"][0]["options"]["A"]
    monkeypatch.setattr(interview, "_call_llm_json", lambda prompt: json.dumps(payload, ensure_ascii=False))
    items = interview.generate_mcq_bank(company="", jd_text="x", resume_text="y")
    assert len(items) == 19  # 重复选项题被丢弃


def test_review_mcq_bank_drops_and_fixes(monkeypatch) -> None:
    verdict = json.dumps({"drop": [2], "fix": [{"index": 1, "answer": "C"}]})
    monkeypatch.setattr(interview, "_call_llm_json", lambda prompt: verdict)
    items = [
        {"question": "q1", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer": "A", "analysis": "x"},
        {"question": "q2", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer": "B", "analysis": "y"},
    ]
    reviewed = interview.review_mcq_bank(items)
    assert len(reviewed) == 1
    assert reviewed[0]["answer"] == "C"


def test_review_mcq_bank_keeps_original_on_failure(monkeypatch) -> None:
    monkeypatch.setattr(interview, "_call_llm_json", lambda prompt: "not json")
    items = [{"question": "q", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer": "A", "analysis": "x"}]
    assert interview.review_mcq_bank(items) == items


def test_generate_qa_bank_chunks_long_requests(monkeypatch) -> None:
    calls = {"n": 0}

    def fake(prompt):
        calls["n"] += 1
        return _valid_qa_payload(count=15)

    monkeypatch.setattr(interview, "_call_llm_json", fake)
    items = interview.generate_qa_bank(company="", jd_text="后端", resume_text="Python")
    assert calls["n"] == 2  # 30 题分两批
    assert len(items) == 15  # 两批同构 mock 去重后剩 15



def test_extract_real_interview_questions(monkeypatch) -> None:
    payload = json.dumps(
        {
            "questions": [
                {"question": "讲讲 MySQL 索引失效的场景", "source_index": 1},
                {"question": "Python GIL 对多线程的影响", "source_index": 1},
                {"question": "讲讲 MySQL 索引失效的场景", "source_index": 1},
            ]
        },
        ensure_ascii=False,
    )
    monkeypatch.setattr(interview, "_call_llm_json", lambda prompt, max_tokens=4000: payload)
    refs = [{"title": "百度后端面经", "url": "https://example.com/1", "snippet": "..."}]
    items = interview.extract_real_interview_questions(
        company="百度",
        position="后端",
        references_text="[1] 百度后端面经\n问了索引和 GIL",
        references=refs,
        limit=20,
    )
    assert len(items) == 2
    assert items[0]["source_title"] == "百度后端面经"
    assert items[0]["source_url"].startswith("https://")


def test_extract_real_interview_questions_empty_without_refs() -> None:
    assert interview.extract_real_interview_questions(
        company="x", position="y", references_text="", references=[], limit=20
    ) == []


def test_generate_qa_bank_on_chunk_callback(monkeypatch) -> None:
    calls = []

    def fake(prompt):
        return _valid_qa_payload(count=15)

    monkeypatch.setattr(interview, "_call_llm_json", fake)
    interview.generate_qa_bank(
        company="",
        jd_text="后端",
        resume_text="Python",
        count=30,
        on_chunk=lambda i, total: calls.append((i, total)),
    )
    assert calls == [(1, 2), (2, 2)]



def test_build_interview_bank_emits_stages(monkeypatch) -> None:
    stages: list[str] = []

    monkeypatch.setattr(
        interview,
        "extract_real_interview_questions",
        lambda **kwargs: [{"question": "真实题1", "source_title": "s", "source_url": "u"}],
    )
    monkeypatch.setattr(
        interview,
        "generate_mcq_bank",
        lambda **kwargs: [
            {
                "question": f"Q{i}",
                "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "answer": "A",
                "analysis": "analysis text here",
            }
            for i in range(10)
        ],
    )
    monkeypatch.setattr(interview, "review_mcq_bank", lambda items, enabled=True: items)

    def fake_qa(**kwargs):
        on_chunk = kwargs.get("on_chunk")
        if on_chunk:
            on_chunk(1, 1)
        return [
            {
                "question": f"简答{i}",
                "category": "基础概念",
                "spoken_answer": "口语答案内容足够长一些",
                "analysis": "题目讲解内容足够长一些文字",
            }
            for i in range(8)
        ]

    monkeypatch.setattr(interview, "generate_qa_bank", fake_qa)

    class DummySession:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            from datetime import datetime

            self.created_at = datetime.utcnow()
            self.questions = []

    class DummyQuestion:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class DummyDB:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def add(self, obj):
            return None

        def commit(self):
            return None

        def refresh(self, obj):
            return None

    import models
    import database

    monkeypatch.setattr(models, "InterviewSession", DummySession)
    monkeypatch.setattr(models, "InterviewQuestion", DummyQuestion)
    monkeypatch.setattr(database, "engine", object())

    # build_interview_bank does: from sqlalchemy.orm import Session as DbSession
    import sqlalchemy.orm as sa_orm

    monkeypatch.setattr(sa_orm, "Session", DummyDB)

    result = interview.build_interview_bank(
        company="百度",
        position="后端",
        jd_text="JD content enough",
        resume_text="简历内容足够长" * 5,
        resume_filename=None,
        user_id="u1",
        references_text="ref text",
        references=[{"title": "t", "url": "u", "snippet": "s"}],
        on_stage=lambda key, msg, prog, extra: stages.append(key),
    )
    assert len(result["mcq"]) == 10
    assert len(result["qa"]) == 8
    assert result["real_questions"][0]["question"] == "真实题1"
    for needed in ("extract", "mcq", "review", "qa", "save", "done"):
        assert needed in stages
