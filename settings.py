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
SPLITTER_VERSION = "article_chunk_v2"

# Model and retrieval settings.
MODEL_NAME = "glm-4.7-flash"
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
CANDIDATE_TOP_K = 8
RETRIEVAL_TOP_K = 3
SCORE_THRESHOLD = 0.3

# PDF chunking settings.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# Evaluation concurrency.
MAX_CONCURRENCY = 3
