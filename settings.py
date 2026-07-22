"""
Project-level configuration.

Keep runtime knobs in one place so local development and deployment
can use the same code with different environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = Path.home() / "Downloads"
DEFAULT_STUDENT_MANUAL_PATH = DOWNLOADS_DIR / "学生手册（教育管理篇）2025版.pdf"
DEFAULT_PDF_PATH = DEFAULT_STUDENT_MANUAL_PATH if DEFAULT_STUDENT_MANUAL_PATH.exists() else Path("student_manual.pdf")


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


# Knowledge source configuration.
PDF_PATH = resolve_path(os.getenv("PDF_PATH", str(DEFAULT_PDF_PATH)))
POLICY_DOCUMENTS = [
    {
        "document_id": "student_manual_education_2025",
        "document_name": "学生手册（教育管理篇）2025版",
        "document_type": "manual",
        "topic": "student_management",
        "authority_level": 70,
        "path": str(PDF_PATH),
    },
    {
        "document_id": "undergraduate_scholarship_rules_2025",
        "document_name": "本科生奖学金评定办法",
        "document_type": "school_policy",
        "topic": "scholarship",
        "authority_level": 95,
        "path": str(DOWNLOADS_DIR / "重邮〔2025〕231 号关于印发《本科生奖学金评定办法》的通知.pdf"),
    },
    {
        "document_id": "comprehensive_evaluation_rules_2025",
        "document_name": "本科生综合素质测评办法",
        "document_type": "evaluation_policy",
        "topic": "comprehensive_evaluation",
        "authority_level": 95,
        "path": str(DOWNLOADS_DIR / "重邮〔2025〕232 号关于印发《本科生综合素质测评办法》的通知.pdf"),
    },
    {
        "document_id": "social_scholarship_rules",
        "document_name": "本科生社会奖学金评定办法",
        "document_type": "special_policy",
        "topic": "scholarship",
        "authority_level": 90,
        "path": str(DOWNLOADS_DIR / "重庆邮电大学本科生社会奖学金评定办法 (1).docx"),
    },
]

# Vector store and index metadata locations.
VECTOR_DB_DIR = Path("chroma_db")
INDEX_META_PATH = Path("manual_index_meta.json")
SPLITTER_VERSION = "parent_child_article_v1"

# Model and retrieval settings.
MODEL_NAME = os.getenv("LLM_MODEL", "deepseek-v4-flash")
# Embedding still uses Zhipu; chat/rewrite/summary use DeepSeek above.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "embedding-2")
# Wider candidate pool for hybrid fusion; final answer still uses RETRIEVAL_TOP_K.
CANDIDATE_TOP_K = int(os.getenv("CANDIDATE_TOP_K", "16"))
HYBRID_FUSION_TOP_K = int(os.getenv("HYBRID_FUSION_TOP_K", "12"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
# Hard char budget for the final evidence block. Page texts median ~682 chars,
# so 4 pages ≈ 2.8k chars; the cap only bites on long-page outliers.
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "4500"))
SCORE_THRESHOLD = 0.3
# Evidence-gate defaults. They must be calibrated against a human-reviewed
# retrieval set before changing production behavior.
RETRIEVAL_VECTOR_SUPPORTED_THRESHOLD = float(os.getenv("RETRIEVAL_VECTOR_SUPPORTED_THRESHOLD", "0.52"))
RETRIEVAL_VECTOR_OUT_OF_SCOPE_THRESHOLD = float(os.getenv("RETRIEVAL_VECTOR_OUT_OF_SCOPE_THRESHOLD", "0.30"))
RETRIEVAL_KEYWORD_COVERAGE_THRESHOLD = float(os.getenv("RETRIEVAL_KEYWORD_COVERAGE_THRESHOLD", "0.50"))
# Strong lexical hit can support even when vector is only mid-range (common for
# short colloquial campus questions whose wording differs from statute text).
RETRIEVAL_BM25_SUPPORTED_THRESHOLD = float(os.getenv("RETRIEVAL_BM25_SUPPORTED_THRESHOLD", "4.0"))
# Min vector score required for the BM25+coverage support path. Calibrated
# 2026-07-22 on 120 gold cases: 0.28 separates campus-flavored-but-unanswerable
# questions (hours/phone/menu, vector<=0.279) from colloquial statute QA.
RETRIEVAL_BM25_VECTOR_FLOOR = float(os.getenv("RETRIEVAL_BM25_VECTOR_FLOOR", "0.28"))
# Knowledge-path rewrite gate. auto = retrieve first, rewrite only when weak.
REWRITE_MODE = os.getenv("REWRITE_MODE", "auto")  # auto | on | off
REWRITE_EXPANSION_WEIGHT = float(os.getenv("REWRITE_EXPANSION_WEIGHT", "0.5"))
# After hybrid hits, pull same-document neighbor pages into the candidate pool
# so cross-page clauses are less likely to be truncated in the final context.
NEIGHBOR_PAGE_RADIUS = int(os.getenv("NEIGHBOR_PAGE_RADIUS", "1"))
# 0 = expand neighbors for the whole hybrid pool (pool itself is bounded by
# HYBRID_FUSION_TOP_K). A fixed seed cap proved fragile: a page ranked one
# slot outside the cap silently loses its clause-bearing neighbor page.
NEIGHBOR_SEED_TOP_N = int(os.getenv("NEIGHBOR_SEED_TOP_N", "0"))

# Working memory: recent turns stay verbatim; older turns collapse to one MySQL summary.
SHORT_TERM_ROUNDS = int(os.getenv("SHORT_TERM_ROUNDS", "6"))
SHORT_TERM_MESSAGE_LIMIT = SHORT_TERM_ROUNDS * 2
PROFILE_INJECT_K = int(os.getenv("PROFILE_INJECT_K", "5"))
OVERFLOW_SUMMARY_ENABLED = os.getenv("OVERFLOW_SUMMARY_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OVERFLOW_SUMMARY_MAX_CHARS = int(os.getenv("OVERFLOW_SUMMARY_MAX_CHARS", "300"))
# How many past messages to load when splitting near/overflow windows.
SESSION_HISTORY_LOAD_LIMIT = int(os.getenv("SESSION_HISTORY_LOAD_LIMIT", "80"))

# Profile memory write gate (P1): regex always-on; LLM extract optional.
MEMORY_LLM_EXTRACT_ENABLED = os.getenv("MEMORY_LLM_EXTRACT_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MEMORY_AUTO_CONFIRM_MIN_CONFIDENCE = float(
    os.getenv("MEMORY_AUTO_CONFIRM_MIN_CONFIDENCE", "0.90")
)
MEMORY_PENDING_MIN_CONFIDENCE = float(os.getenv("MEMORY_PENDING_MIN_CONFIDENCE", "0.70"))
MEMORY_CANDIDATE_TTL_HOURS = int(os.getenv("MEMORY_CANDIDATE_TTL_HOURS", "72"))
# Follow-up resolution: LLM rewrites deictic follow-ups into one clean
# standalone query; falls back to rule-based concatenation on failure.
FOLLOWUP_LLM_RESOLVE = os.getenv("FOLLOWUP_LLM_RESOLVE", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# When false, skip injecting legacy Chroma session summary_window into prompt
# (MySQL overflow summary already covers working-memory overflow).
CHROMA_SESSION_SUMMARY_IN_PROMPT = os.getenv(
    "CHROMA_SESSION_SUMMARY_IN_PROMPT", "false"
).lower() in {"1", "true", "yes", "on"}

# PDF chunking settings.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# Evaluation concurrency.
MAX_CONCURRENCY = 3
