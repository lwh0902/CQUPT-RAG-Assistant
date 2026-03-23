"""
向量库管理模块。

这里专门负责：
1. 计算 PDF 指纹，判断文档是否发生变化
2. 决定是首次构建还是直接加载已有向量库
3. 读写索引元数据
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from pdf_loader import load_pdf_documents, split_documents
from settings import (
    CANDIDATE_TOP_K,
    EMBEDDING_MODEL_NAME,
    INDEX_META_PATH,
    PDF_PATH,
    RETRIEVAL_TOP_K,
    SCORE_THRESHOLD,
    SPLITTER_VERSION,
    VECTOR_DB_DIR,
)


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    创建统一的中文 Embedding 模型实例。
    """
    # 先优先尝试只用本地缓存加载。
    # 这样第一次下载完成后，后续启动就不需要再访问 Hugging Face。
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        try:
            return HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            raise RuntimeError(
                "Embedding 模型加载失败。首次运行请确保可以联网下载模型，"
                "后续运行会优先使用本地缓存。"
            ) from exc


def get_pdf_fingerprint(pdf_path: Path) -> str:
    """
    计算 PDF 文件的 SHA256 指纹。

    只要文件内容变化，指纹就会变化。
    """
    sha256 = hashlib.sha256()
    with pdf_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_index_meta(meta_path: Path = INDEX_META_PATH) -> dict:
    """
    读取索引元数据。

    如果文件不存在，返回空字典。
    """
    if not meta_path.exists():
        return {}

    with meta_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_index_meta(meta: dict, meta_path: Path = INDEX_META_PATH) -> None:
    """
    保存索引元数据，供后续判断是否重建向量库。
    """
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)


def is_vector_db_ready(persist_dir: Path = VECTOR_DB_DIR) -> bool:
    """
    判断当前目录下是否存在可加载的 Chroma 数据。
    """
    return (persist_dir / "chroma.sqlite3").exists()


def should_rebuild_vector_db(
    pdf_path: Path = PDF_PATH,
    persist_dir: Path = VECTOR_DB_DIR,
    meta_path: Path = INDEX_META_PATH,
) -> tuple[bool, dict]:
    """
    判断是否需要重建向量库。

    返回值：
    - 第一个值：是否需要重建
    - 第二个值：已有元数据，后续可复用
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到学生手册 PDF：{pdf_path}")

    if not is_vector_db_ready(persist_dir):
        return True, {}

    meta = load_index_meta(meta_path)
    if not meta:
        return True, {}

    current_fingerprint = get_pdf_fingerprint(pdf_path)
    saved_fingerprint = meta.get("pdf_fingerprint")
    saved_path = meta.get("pdf_path")
    saved_splitter_version = meta.get("splitter_version")

    if current_fingerprint != saved_fingerprint:
        return True, meta

    if saved_path != str(pdf_path):
        return True, meta

    if saved_splitter_version != SPLITTER_VERSION:
        return True, meta

    return False, meta


def build_vector_store(
    pdf_path: Path = PDF_PATH,
    persist_dir: Path = VECTOR_DB_DIR,
    meta_path: Path = INDEX_META_PATH,
) -> dict:
    """
    从 PDF 重新构建向量库，并返回构建信息。
    """
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    documents = load_pdf_documents(pdf_path)
    chunks = split_documents(documents)
    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )

    # 某些版本的 Chroma 仍保留 persist 方法，这里做兼容调用。
    if hasattr(vector_store, "persist"):
        vector_store.persist()

    fingerprint = get_pdf_fingerprint(pdf_path)
    meta = {
        "pdf_path": str(pdf_path),
        "pdf_fingerprint": fingerprint,
        "splitter_version": SPLITTER_VERSION,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "page_count": len(documents),
        "chunk_count": len(chunks),
    }
    save_index_meta(meta, meta_path)

    return meta


def load_vector_store(persist_dir: Path = VECTOR_DB_DIR) -> Chroma:
    """
    直接加载已有的 Chroma 向量库。
    """
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def create_or_load_retriever(
    pdf_path: Path = PDF_PATH,
    persist_dir: Path = VECTOR_DB_DIR,
    meta_path: Path = INDEX_META_PATH,
):
    """
    创建或加载 retriever，并返回初始化信息。

    返回：
    - retriever：用于问答检索
    - init_info：用于前端提示当前是“首次构建”还是“直接加载”
    """
    need_rebuild, old_meta = should_rebuild_vector_db(pdf_path, persist_dir, meta_path)

    if need_rebuild:
        meta = build_vector_store(pdf_path, persist_dir, meta_path)
        status = "rebuilt"
        message = "已根据学生手册 PDF 重新构建向量库。"
    else:
        meta = old_meta
        status = "loaded"
        message = "检测到学生手册未变化，已直接加载已有向量库。"

    vector_store = load_vector_store(persist_dir)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": CANDIDATE_TOP_K, "score_threshold": SCORE_THRESHOLD},
    )

    init_info = {
        "status": status,
        "message": message,
        "pdf_path": str(pdf_path),
        "page_count": meta.get("page_count"),
        "chunk_count": meta.get("chunk_count"),
        "updated_at": meta.get("updated_at"),
    }
    return retriever, init_info
