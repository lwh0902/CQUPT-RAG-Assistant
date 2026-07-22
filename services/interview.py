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
2. analysis 用 2-3 句话讲清考点与判断依据；
3. 严格输出 JSON，不要输出任何其他文字，格式：
{{"questions":[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"answer":"A","analysis":"..."}}]}}

【目标公司】{company}
【岗位JD】
{jd}

【候选人简历】
{resume}
"""

_QA_PROMPT = """你是一名资深技术面试官。根据【岗位JD】与【候选人简历】，生成 {count} 道高频面试简答题（项目深挖、技术原理、场景设计均可）。

要求：
1. spoken_answer 是 20 秒口语化回答：自然说话风格、突出要点，约 60-90 字，不要书面腔；
2. analysis 是答案解析：面试官考察点 + 答题要点拆解，3-5 句话；
3. 严格输出 JSON，不要输出任何其他文字，格式：
{{"questions":[{{"question":"...","spoken_answer":"...","analysis":"..."}}]}}

【目标公司】{company}
【岗位JD】
{jd}

【候选人简历】
{resume}
"""


def _call_llm_json(prompt: str) -> str:
    response = get_llm_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        timeout=120,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return (response.choices[0].message.content or "").strip()


def _strip_json_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _generate_bank(prompt: str, schema: type[McqBank] | type[QaBank], expected: int) -> list[dict[str, Any]]:
    """Call LLM, validate JSON schema, retry once on failure."""
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            raw = _strip_json_fence(_call_llm_json(prompt))
            payload = schema.model_validate(json.loads(raw))
            items = [item.model_dump() for item in payload.questions]
            if len(items) < max(8, expected - 4):
                raise ValueError(f"题目数量不足：{len(items)}")
            return items[:expected]
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            logger.warning("interview bank generation attempt %d failed: %s", attempt + 1, exc)
    raise ValueError(f"题库生成失败，请重试（{last_error}）")


def generate_mcq_bank(*, company: str, jd_text: str, resume_text: str, count: int = 20) -> list[dict[str, Any]]:
    prompt = _MCQ_PROMPT.format(
        count=count,
        company=company or "未指定",
        jd=clamp_text(jd_text, MAX_JD_CHARS),
        resume=clamp_text(resume_text, MAX_RESUME_CHARS),
    )
    return _generate_bank(prompt, McqBank, count)


def generate_qa_bank(*, company: str, jd_text: str, resume_text: str, count: int = 20) -> list[dict[str, Any]]:
    prompt = _QA_PROMPT.format(
        count=count,
        company=company or "未指定",
        jd=clamp_text(jd_text, MAX_JD_CHARS),
        resume=clamp_text(resume_text, MAX_RESUME_CHARS),
    )
    return _generate_bank(prompt, QaBank, count)
