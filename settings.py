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
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
SCORE_THRESHOLD = 0.3
# Evidence-gate defaults. They must be calibrated against a human-reviewed
# retrieval set before changing production behavior.
RETRIEVAL_VECTOR_SUPPORTED_THRESHOLD = float(os.getenv("RETRIEVAL_VECTOR_SUPPORTED_THRESHOLD", "0.52"))
RETRIEVAL_VECTOR_OUT_OF_SCOPE_THRESHOLD = float(os.getenv("RETRIEVAL_VECTOR_OUT_OF_SCOPE_THRESHOLD", "0.30"))
RETRIEVAL_KEYWORD_COVERAGE_THRESHOLD = float(os.getenv("RETRIEVAL_KEYWORD_COVERAGE_THRESHOLD", "0.50"))
# Strong lexical hit can support even when vector is only mid-range (common for
# short colloquial campus questions whose wording differs from statute text).
RETRIEVAL_BM25_SUPPORTED_THRESHOLD = float(os.getenv("RETRIEVAL_BM25_SUPPORTED_THRESHOLD", "4.0"))
# Knowledge-path rewrite gate. auto = retrieve first, rewrite only when weak.
REWRITE_MODE = os.getenv("REWRITE_MODE", "auto")  # auto | on | off
REWRITE_EXPANSION_WEIGHT = float(os.getenv("REWRITE_EXPANSION_WEIGHT", "0.5"))
# After hybrid hits, pull same-document neighbor pages into the candidate pool
# so cross-page clauses are less likely to be truncated in the final context.
NEIGHBOR_PAGE_RADIUS = int(os.getenv("NEIGHBOR_PAGE_RADIUS", "1"))
NEIGHBOR_SEED_TOP_N = int(os.getenv("NEIGHBOR_SEED_TOP_N", "5"))

# PDF chunking settings.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# Evaluation concurrency.
MAX_CONCURRENCY = 3
