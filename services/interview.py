"""Interview assistant: resume text extraction + LLM question-bank generation.

Design rules:
- Resume text comes from PDF text-layer (pypdf) or pasted text — no multimodal.
- Generation is strict-JSON with schema validation and one retry; banks are
  persisted so users can revisit and export them.
"""

from __future__ import annotations

import io
import json
import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from services.llm import get_llm_client
from settings import MODEL_NAME

logger = logging.getLogger(__name__)

MAX_RESUME_CHARS = 6000
MAX_JD_CHARS = 4000


# ---------------------------------------------------------------------------
# Resume text extraction
# ---------------------------------------------------------------------------

def _extract_docx_text(file_bytes: bytes) -> str:
    """Extract text from a .docx upload via stdlib zip + XML (no extra deps)."""
    import zipfile
    from xml.etree import ElementTree

    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            xml_text = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("DOCX 简历解析失败，请确认文件未损坏且确为 .docx 格式") from exc

    root = ElementTree.fromstring(xml_text)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace) if node.text]
        paragraph_text = "".join(texts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)
    text = "\n".join(paragraphs).strip()
    if not text:
        raise ValueError("该 DOCX 没有可提取的文字内容，请改用文本粘贴")
    return text


def extract_resume_text(file_bytes: bytes, filename: str) -> str:
    """Extract resume text from PDF (text layer), DOCX, or plain-text upload."""
    name = (filename or "").lower()
    if name.endswith(".docx") or name.endswith(".doc"):
        return _extract_docx_text(file_bytes)
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ValueError("服务器缺少 pypdf，无法解析 PDF 简历") from exc
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as exc:
            raise ValueError("PDF 简历解析失败，请检查文件是否损坏") from exc
        parts = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(parts).strip()
        if not text:
            raise ValueError("该 PDF 没有可提取的文字层（可能是扫描件），请改用文本粘贴")
        return text
    try:
        return file_bytes.decode("utf-8", errors="ignore").strip()
    except Exception as exc:
        raise ValueError("简历内容读取失败，请上传 PDF/DOCX 或直接粘贴文本") from exc


def clamp_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Question-bank schemas
# ---------------------------------------------------------------------------

class McqItem(BaseModel):
    question: str = Field(min_length=4, max_length=400)
    options: dict[str, str]
    answer: str = Field(pattern="^[ABCD]$")
    analysis: str = Field(min_length=4, max_length=800)


class QaItem(BaseModel):
    question: str = Field(min_length=4, max_length=400)
    category: str = Field(default="综合", max_length=20)
    spoken_answer: str = Field(min_length=10, max_length=600)
    analysis: str = Field(min_length=4, max_length=1000)


class McqBank(BaseModel):
    questions: list[McqItem]


class QaBank(BaseModel):
    questions: list[QaItem]


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

_MCQ_PROMPT = """你是一名资深技术面试官。根据【岗位JD】与【候选人简历】，生成 {count} 道单项选择题，考察候选人对简历中技术栈与项目细节的掌握（以概念题为主，例如 Python GIL、数据库索引、框架原理等），不要泛泛考“自我介绍”。

要求：
1. 题目必须与简历/JD 中的技术点强相关，选项有区分度，只有一个正确答案；
2. 四个选项必须互不相同，禁止同义改写、禁止近似重复（例如同一行代码出现两次）；
3. 全题有且仅有一个正确陈述——若某选项“不算错但不是最佳”，不要放进选项；
4. analysis 用 2-3 句话讲清考点与判断依据，并说明其余选项为什么错；
5. 严格输出 JSON，不要输出任何其他文字，格式：
{{"questions":[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"answer":"A","analysis":"..."}}]}}

【目标公司】{company}
【目标岗位】{position}
【岗位JD】
{jd}

【候选人简历】
{resume}
{references_block}"""

_QA_PROMPT = """你是一名资深技术面试官。根据【岗位JD】与【候选人简历】，生成 {count} 道高频面试简答题。

要求：
1. 每题标注 category，只能是：项目深挖 / 编程语言 / 架构设计 / 基础概念 / 行为面试；
2. 题型分布尽量均衡，项目深挖结合候选人简历真实项目；
3. spoken_answer 是 40-60 秒的完整口语回答：自然说话风格，约 150-220 字，
   有结构地展开（背景一句话 → 要点 2-3 条 → 结果/收获一句话），不要书面腔，也不要只给一两句话；
4. analysis 是「题目讲解」，不是答案解析：
   - 面试官为什么问这道题（考察点与背景知识讲清楚）；
   - 这道题该怎么理解、从哪里切入思考（思路框架）；
   - 关联的核心知识点是什么（让用户看懂后自己也能答，而不是死记答案）；
   - 4-6 句话，像老师讲课一样把题讲透；
5. 严格输出 JSON，不要输出任何其他文字，格式：
{{"questions":[{{"question":"...","category":"项目深挖","spoken_answer":"...","analysis":"..."}}]}}

【目标公司】{company}
【目标岗位】{position}
【岗位JD】
{jd}

【候选人简历】
{resume}
{references_block}"""

_TARGETED_MCQ_PROMPT = """你是一名资深技术面试官。候选人上一轮选择题在以下知识点上表现薄弱：
【薄弱知识点】
{weak_points}

请围绕这些薄弱知识点（可结合简历与岗位），生成 {count} 道单项选择题进行针对性巩固。

要求：
1. 题目聚焦薄弱点，选项有区分度，只有一个正确答案；
2. 四个选项必须互不相同，禁止同义改写、禁止近似重复；
3. 全题有且仅有一个正确陈述；
4. analysis 用 2-3 句话讲清考点与判断依据；
5. 严格输出 JSON，不要输出任何其他文字，格式：
{{"questions":[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"answer":"A","analysis":"..."}}]}}

【目标公司】{company}
【目标岗位】{position}
【岗位JD】
{jd}

【候选人简历】
{resume}
"""

_REPORT_PROMPT = """你是一名面试教练。候选人在模拟选择题中答错了以下题目：

{wrong_block}

请输出一份 Markdown 格式的薄弱点分析报告：
1. 「薄弱知识点」：按主题归纳（不超过 5 个主题），每个主题列出答错的相关题目；
2. 「问题诊断」：每个主题一句话分析可能的理解误区；
3. 「复习建议」：每个主题给出 1-2 条具体可执行的复习建议（看什么、练什么）；
4. 最后一段「总体建议」：2-3 句话。
只输出 Markdown 正文，不要输出其他内容。
"""


def _call_llm_json(prompt: str, *, max_tokens: int = 16000) -> str:
    response = get_llm_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        timeout=180,
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return (response.choices[0].message.content or "").strip()


def _strip_json_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _normalize_option(text: str) -> str:
    return re.sub(r"[\s，。；：、,.;:'\"()（）\[\]{}]+", "", (text or "").lower())


def _options_are_suspicious(options: dict[str, str]) -> bool:
    """Detect duplicate / near-duplicate options (e.g. same code line twice)."""
    normalized = [_normalize_option(value) for value in options.values()]
    normalized = [value for value in normalized if value]
    if len(normalized) < 4 or len(set(normalized)) < len(normalized):
        return True
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            a, b = normalized[i], normalized[j]
            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            if shorter and shorter in longer and len(shorter) / len(longer) >= 0.8:
                return True
    return False


def _generate_bank(prompt: str, schema: type[McqBank] | type[QaBank], expected: int) -> list[dict[str, Any]]:
    """Call LLM, validate JSON schema, retry once on failure."""
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            raw = _strip_json_fence(_call_llm_json(prompt))
            payload = schema.model_validate(json.loads(raw))
            items = [item.model_dump() for item in payload.questions]
            if schema is McqBank:
                items = [item for item in items if not _options_are_suspicious(item.get("options") or {})]
            if len(items) < max(8, expected - 4):
                raise ValueError(f"题目数量不足：{len(items)}")
            return items[:expected]
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            logger.warning("interview bank generation attempt %d failed: %s", attempt + 1, exc)
    raise ValueError(f"题库生成失败，请重试（{last_error}）")


# ---------------------------------------------------------------------------
# LLM self-review pass: catch wrong answer keys / ambiguous questions
# ---------------------------------------------------------------------------

_REVIEW_PROMPT = """你是面试题质检员。请逐题审查以下单项选择题，找出有问题的题目：
- 标答本身错误；
- 存在两个及以上正确/不算错的选项；
- 选项重复或近似重复；
- 题目表述有歧义。

只输出 JSON，格式：{{"drop": [题号1, 题号2], "fix": [{{"index": 题号, "answer": "修正后的正确选项字母"}}]}}
没有问题时输出 {{"drop": [], "fix": []}}。题号从 1 开始。

【题目】
{questions_block}
"""


def review_mcq_bank(items: list[dict[str, Any]], *, enabled: bool = True) -> list[dict[str, Any]]:
    """Second-pass LLM review: drop or fix questions with bad answer keys."""
    if not enabled or not items:
        return items
    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        options = " ".join(f"{k}.{v}" for k, v in (item.get("options") or {}).items())
        lines.append(f"{index}. {item.get('question', '')}\n   {options}\n   标答：{item.get('answer', '')}")
    prompt = _REVIEW_PROMPT.format(questions_block="\n".join(lines))
    try:
        raw = _strip_json_fence(_call_llm_json(prompt))
        verdict = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("mcq review pass failed; keeping original bank", exc_info=True)
        return items

    drop = {int(i) for i in verdict.get("drop", []) if isinstance(i, int)}
    fixes: dict[int, str] = {}
    for fix in verdict.get("fix", []):
        try:
            index = int(fix.get("index"))
            answer = str(fix.get("answer", "")).strip().upper()
            if answer in {"A", "B", "C", "D"}:
                fixes[index] = answer
        except (TypeError, ValueError, AttributeError):
            continue

    reviewed: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if index in drop:
            continue
        if index in fixes:
            item = {**item, "answer": fixes[index]}
        reviewed.append(item)
    return reviewed


def generate_mcq_bank(*, company: str, position: str = "", jd_text: str, resume_text: str, references: str = "", count: int = 20) -> list[dict[str, Any]]:
    prompt = _MCQ_PROMPT.format(
        count=count,
        company=company or "未指定",
        position=position or "未指定",
        jd=clamp_text(jd_text, MAX_JD_CHARS),
        resume=clamp_text(resume_text, MAX_RESUME_CHARS),
        references_block=_references_block(references),
    )
    return _generate_bank(prompt, McqBank, count)


def generate_qa_bank(
    *,
    company: str,
    position: str = "",
    jd_text: str,
    resume_text: str,
    references: str = "",
    count: int = 30,
    on_chunk: Any = None,
) -> list[dict[str, Any]]:
    """Generate the QA bank in chunks: 30 long-form items exceed the model's
    output budget in one call (truncated JSON = unterminated string).

    on_chunk(index_1based, total_chunks) is invoked before each chunk call so
    SSE progress can surface real stage text.
    """
    chunk_size = 15
    total_chunks = max(1, (count + chunk_size - 1) // chunk_size)

    def _one(chunk: int, chunk_index: int) -> list[dict[str, Any]]:
        if on_chunk is not None:
            on_chunk(chunk_index, total_chunks)
        prompt = _QA_PROMPT.format(
            count=chunk,
            company=company or "未指定",
            position=position or "未指定",
            jd=clamp_text(jd_text, MAX_JD_CHARS),
            resume=clamp_text(resume_text, MAX_RESUME_CHARS),
            references_block=_references_block(references),
        )
        return _generate_bank(prompt, QaBank, chunk)

    if count <= chunk_size:
        return _one(count, 1)

    merged: list[dict[str, Any]] = []
    remaining = count
    chunk_index = 0
    while remaining > 0:
        chunk = min(chunk_size, remaining)
        chunk_index += 1
        merged.extend(_one(chunk, chunk_index))
        remaining -= chunk
    # Deduplicate near-identical questions across chunks.
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in merged:
        key = re.sub(r"\W+", "", item.get("question", ""))[:30]
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:count]


def generate_targeted_mcq_bank(*, company: str, position: str, jd_text: str, resume_text: str, weak_points: str, count: int = 10) -> list[dict[str, Any]]:
    prompt = _TARGETED_MCQ_PROMPT.format(
        count=count,
        company=company or "未指定",
        position=position or "未指定",
        jd=clamp_text(jd_text, MAX_JD_CHARS),
        resume=clamp_text(resume_text, MAX_RESUME_CHARS),
        weak_points=clamp_text(weak_points, 1500),
    )
    return _generate_bank(prompt, McqBank, count)


def generate_weakness_report(wrong_items: list[dict[str, Any]]) -> str:
    """LLM markdown report over the wrongly-answered MCQ items."""
    if not wrong_items:
        return ""
    lines: list[str] = []
    for index, item in enumerate(wrong_items, start=1):
        options = " ".join(f"{k}.{v}" for k, v in (item.get("options") or {}).items())
        lines.append(
            f"{index}. {item.get('question', '')}\n   选项：{options}\n   正确答案：{item.get('answer', '')}"
        )
    prompt = _REPORT_PROMPT.format(wrong_block="\n".join(lines))
    raw = _call_llm_json(prompt)  # free-form markdown, not JSON
    return raw.strip()


# ---------------------------------------------------------------------------
# Interview-reference web search (面经)
# ---------------------------------------------------------------------------

async def search_interview_references(company: str, position: str, *, limit: int = 6) -> tuple[str, list[dict[str, Any]]]:
    """Search the web for interview experiences (面经) for company+position.

    Graceful fallback: returns ("", []) when no search provider is configured.
    Each ref: {title, url, snippet}.
    """
    company = (company or "").strip()
    position = (position or "").strip()
    if not company and not position:
        return "", []
    query = " ".join(part for part in (company, position) if part) + " 面试 面经"
    try:
        from services.web_search import get_web_search_provider

        provider = get_web_search_provider()
        sources = await provider.search(query, limit=limit)
    except Exception:
        logger.warning("interview reference search failed", exc_info=True)
        return "", []

    snippets: list[str] = []
    refs: list[dict[str, Any]] = []
    for index, source in enumerate(sources[:limit], start=1):
        snippet = " ".join((source.snippet or "").split())[:400]
        title = (source.title or "").strip()
        url = (source.url or "").strip()
        if not snippet and not title:
            continue
        snippets.append(f"[{index}] {title}\n{snippet}")
        refs.append({"title": title, "url": url, "snippet": snippet})
    if not snippets:
        return "", []
    return "\n\n".join(snippets), refs


_EXTRACT_REAL_QUESTIONS_PROMPT = """你是面试题整理助手。根据下面【面经搜索结果】，提炼最多 {limit} 道真实面试题题干。

规则：
1. 只输出题干，不要答案、不要解析；
2. 优先技术题 / 项目题 / 场景题；过滤广告、灌水、纯吐槽；
3. 题干简洁完整，像候选人可能被问到的原话；
4. 去重；不够就少出，不要编造搜索结果里没有的题；
5. 严格 JSON：{{"questions":[{{"question":"...","source_index":1}}]}}
   source_index 对应面经条目编号（从 1 起），不确定就填 1。

【目标公司】{company}
【目标岗位】{position}
【面经搜索结果】
{references}
"""


class RealQuestionItem(BaseModel):
    question: str = Field(min_length=4, max_length=300)
    source_index: int = Field(default=1, ge=1, le=20)


class RealQuestionBank(BaseModel):
    questions: list[RealQuestionItem]


def extract_real_interview_questions(
    *,
    company: str,
    position: str,
    references_text: str,
    references: list[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Turn raw 面经 snippets into a transparent list of real interview questions (no answers)."""
    references_text = (references_text or "").strip()
    if not references_text:
        return []
    prompt = _EXTRACT_REAL_QUESTIONS_PROMPT.format(
        limit=limit,
        company=company or "未指定",
        position=position or "未指定",
        references=clamp_text(references_text, 6000),
    )
    try:
        raw = _strip_json_fence(_call_llm_json(prompt, max_tokens=4000))
        payload = RealQuestionBank.model_validate(json.loads(raw))
    except Exception:
        logger.warning("extract real interview questions failed", exc_info=True)
        return []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload.questions:
        q = (item.question or "").strip()
        key = re.sub(r"\W+", "", q)[:40]
        if not q or key in seen:
            continue
        seen.add(key)
        src = references[item.source_index - 1] if 1 <= item.source_index <= len(references) else (references[0] if references else {})
        out.append(
            {
                "question": q,
                "source_title": (src or {}).get("title") or "",
                "source_url": (src or {}).get("url") or "",
            }
        )
        if len(out) >= limit:
            break
    return out


_TUTOR_SYSTEM = """你是「面试小助教」，待在面试助手页面里帮用户解惑。

风格：
- 用大白话 + 必要的专业术语，把知识点讲明白；
- 先给结论，再给 2-4 个要点，必要时举一个短例子；
- 不要长篇八股，不要堆砌清单；中文回答，简洁有用；
- 用户可能会粘贴面试题或某个概念，直接针对内容讲解；
- 不要主动要求用户提供简历/JD，除非对方主动提到。
"""


async def stream_interview_tutor(messages: list[dict[str, str]]):
    """Stream a lightweight tutor reply (no tools, no RAG, no auto question context)."""
    from services.llm import stream_llm_text

    cleaned: list[dict[str, str]] = []
    for msg in messages[-12:]:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content[:2000]})
    if not cleaned or cleaned[-1]["role"] != "user":
        raise ValueError("请先输入你想问的问题")

    full = [{"role": "system", "content": _TUTOR_SYSTEM}, *cleaned]
    async for content in stream_llm_text(full, temperature=0.5):
        yield "token", content
    yield "done", ""


def _references_block(references: str) -> str:
    references = (references or "").strip()
    if not references:
        return ""
    return f"\n【面经参考】（来自联网搜索的真实面试经验，出题时优先参考其中的高频考点）\n{references}\n"


# ---------------------------------------------------------------------------
# Full bank generation with progress callbacks (for SSE)
# ---------------------------------------------------------------------------

STAGE_PROGRESS = {
    "parse": 4,
    "search": 12,
    "extract": 22,
    "mcq": 45,
    "review": 58,
    "qa": 88,
    "save": 96,
    "done": 100,
}


def build_interview_bank(
    *,
    company: str,
    position: str,
    jd_text: str,
    resume_text: str,
    resume_filename: str | None,
    user_id: str,
    references_text: str,
    references: list[dict[str, Any]],
    on_stage: Any = None,
) -> dict[str, Any]:
    """Synchronous bank build used by the SSE generator endpoint."""
    import uuid

    from database import engine
    from models import InterviewQuestion, InterviewSession
    from settings import INTERVIEW_MCQ_REVIEW
    from sqlalchemy.orm import Session as DbSession

    def stage(key: str, message: str, **extra: Any) -> None:
        if on_stage is not None:
            on_stage(key, message, STAGE_PROGRESS.get(key, 0), extra)

    stage("extract", "正在从面经里提炼真实面试题…" if references else "未搜到面经，跳过真题提炼…")
    real_questions = extract_real_interview_questions(
        company=company,
        position=position,
        references_text=references_text,
        references=references,
        limit=20,
    ) if references_text else []

    stage("mcq", "正在结合 JD 与简历设计选择题…")
    mcq_raw = generate_mcq_bank(
        company=company,
        position=position,
        jd_text=jd_text,
        resume_text=resume_text,
        references=references_text,
    )

    stage("review", "正在质检选择题答案…")
    mcq = review_mcq_bank(mcq_raw, enabled=INTERVIEW_MCQ_REVIEW)

    def on_qa_chunk(index: int, total: int) -> None:
        stage(
            "qa",
            f"正在撰写简答题（第 {index}/{total} 批）…",
            chunk=index,
            total_chunks=total,
        )

    qa = generate_qa_bank(
        company=company,
        position=position,
        jd_text=jd_text,
        resume_text=resume_text,
        references=references_text,
        on_chunk=on_qa_chunk,
    )

    stage("save", "正在保存题库…")
    with DbSession(engine) as db:
        session = InterviewSession(
            id=str(uuid.uuid4()),
            user_id=user_id,
            company=(company or "").strip()[:100],
            position=(position or "").strip()[:100],
            jd_text=jd_text,
            resume_text=resume_text,
            resume_filename=resume_filename,
            reference_used=bool(references),
            reference_json=json.dumps(
                {"sources": references, "real_questions": real_questions},
                ensure_ascii=False,
            )
            if (references or real_questions)
            else None,
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

        data = {
            "id": session.id,
            "company": session.company,
            "position": getattr(session, "position", "") or "",
            "jd_text": session.jd_text,
            "resume_text": session.resume_text,
            "resume_filename": session.resume_filename,
            "reference_used": bool(getattr(session, "reference_used", False)),
            "references": references,
            "real_questions": real_questions,
            "report_text": None,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "mcq": [{**item, "round": 1} for item in mcq],
            "qa": qa,
        }
    stage("done", "题库生成完成", session_id=data["id"])
    return data
