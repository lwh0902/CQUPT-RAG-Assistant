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
DEFAULT_PDF_PATH = Path(r"C:\Users\皓\Downloads\学生手册（教育管理篇）2025版.pdf")


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


# Knowledge source configuration.
PDF_PATH = resolve_path(os.getenv("PDF_PATH", str(DEFAULT_PDF_PATH)))

# Vector store and index metadata locations.
VECTOR_DB_DIR = Path("chroma_db")
INDEX_META_PATH = Path("manual_index_meta.json")
SPLITTER_VERSION = "article_chunk_v2"

# Model and retrieval settings.
MODEL_NAME = "glm-4"
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
CANDIDATE_TOP_K = 8
RETRIEVAL_TOP_K = 3
SCORE_THRESHOLD = 0.3

# PDF chunking settings.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# Evaluation concurrency.
MAX_CONCURRENCY = 3
