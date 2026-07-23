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


def generate_qa_bank(*, company: str, position: str = "", jd_text: str, resume_text: str, references: str = "", count: int = 30) -> list[dict[str, Any]]:
    """Generate the QA bank in chunks: 30 long-form items exceed the model's
    output budget in one call (truncated JSON = unterminated string)."""
    chunk_size = 15
    if count <= chunk_size:
        prompt = _QA_PROMPT.format(
            count=count,
            company=company or "未指定",
            position=position or "未指定",
            jd=clamp_text(jd_text, MAX_JD_CHARS),
            resume=clamp_text(resume_text, MAX_RESUME_CHARS),
            references_block=_references_block(references),
        )
        return _generate_bank(prompt, QaBank, count)

    merged: list[dict[str, Any]] = []
    remaining = count
    while remaining > 0:
        chunk = min(chunk_size, remaining)
        prompt = _QA_PROMPT.format(
            count=chunk,
            company=company or "未指定",
            position=position or "未指定",
            jd=clamp_text(jd_text, MAX_JD_CHARS),
            resume=clamp_text(resume_text, MAX_RESUME_CHARS),
            references_block=_references_block(references),
        )
        merged.extend(_generate_bank(prompt, QaBank, chunk))
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

async def search_interview_references(company: str, position: str, *, limit: int = 4) -> tuple[str, list[dict[str, Any]]]:
    """Search the web for interview experiences (面经) for company+position.

    Graceful fallback: returns ("", []) when no search provider is configured.
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
        snippet = " ".join((source.snippet or "").split())[:300]
        title = (source.title or "").strip()
        url = (source.url or "").strip()
        if not snippet:
            continue
        snippets.append(f"[{index}] {title}\n{snippet}")
        refs.append({"title": title, "url": url})
    if not snippets:
        return "", []
    return "\n\n".join(snippets), refs


def _references_block(references: str) -> str:
    references = (references or "").strip()
    if not references:
        return ""
    return f"\n【面经参考】（来自联网搜索的真实面试经验，出题时优先参考其中的高频考点）\n{references}\n"
