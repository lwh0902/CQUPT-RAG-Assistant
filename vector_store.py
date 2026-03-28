from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import CANDIDATE_TOP_K, PDF_PATH


# 向量库固定落盘目录。
PERSIST_DIRECTORY = Path("./chroma_db")

# Chroma 固定集合名，避免不同业务混到同一个集合里。
COLLECTION_NAME = "cqupt_student_manual"

# 轻量元数据文件，用于给上层返回初始化提示信息。
INDEX_META_PATH = Path("manual_index_meta.json")

# 这里保守一些，避免批量过大触发接口参数限制。
EMBEDDING_BATCH_SIZE = 16


load_dotenv()


class BatchedZhipuAIEmbeddings(ZhipuAIEmbeddings):
    """按批次调用 embedding 接口，避免单次输入超过上限。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            embeddings.extend(super().embed_documents(batch))
        return embeddings


def get_embeddings() -> ZhipuAIEmbeddings:
    """创建智谱 Embedding 模型实例。"""
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise ValueError("未找到 ZHIPU_API_KEY，请检查 .env 配置。")
    return BatchedZhipuAIEmbeddings(
        model="embedding-2",
        api_key=api_key,
    )


def load_pdf_documents(pdf_path: Path):
    """优先使用 PyMuPDFLoader 读取 PDF，失败时退回到 PyPDFLoader。"""
    if not pdf_path.exists():
        raise FileNotFoundError(f"未找到 PDF 文件：{pdf_path}")

    try:
        return PyMuPDFLoader(str(pdf_path)).load()
    except Exception:
        return PyPDFLoader(str(pdf_path)).load()


def split_documents(documents):
    """使用递归切分器对 PDF 文档做语义友好的分块。"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    return text_splitter.split_documents(documents)


def save_index_meta(meta: dict) -> None:
    """保存当前向量库的基础元数据。"""
    with INDEX_META_PATH.open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)


def load_index_meta() -> dict:
    """读取元数据文件，不存在时返回空字典。"""
    if not INDEX_META_PATH.exists():
        return {}

    with INDEX_META_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def is_vector_store_ready(persist_dir: Path = PERSIST_DIRECTORY) -> bool:
    """判断本地 Chroma 向量库是否已经存在且集合内确实有数据。"""
    if not persist_dir.exists():
        return False

    sqlite_file = persist_dir / "chroma.sqlite3"
    if not sqlite_file.exists():
        return False

    try:
        vector_store = Chroma(
            persist_directory=str(persist_dir),
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
        )
        return vector_store._collection.count() > 0
    except Exception:
        return False


def clear_existing_collection(persist_dir: Path = PERSIST_DIRECTORY) -> None:
    """建新库前先彻底清理旧集合，防止重复执行脚本导致文档叠加污染。"""
    if not persist_dir.exists():
        return

    # 先尝试从 Chroma 客户端层面删除同名集合。
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(persist_dir))
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
    except Exception:
        pass

    # 再清空整个持久化目录，确保不会残留旧索引文件。
    shutil.rmtree(persist_dir, ignore_errors=True)


def build_vector_store(
    pdf_path: Path = PDF_PATH,
    persist_dir: Path = PERSIST_DIRECTORY,
) -> dict:
    """从本地 PDF 重建 Chroma 向量库，并返回建库信息。"""
    clear_existing_collection(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    documents = load_pdf_documents(pdf_path)
    chunks = split_documents(documents)
    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=COLLECTION_NAME,
    )

    # 兼容旧版本 Chroma 的 persist 行为。
    if hasattr(vector_store, "persist"):
        vector_store.persist()

    meta = {
        "status": "rebuilt",
        "message": "已根据本地 PDF 重建 Chroma 向量库。",
        "pdf_path": str(pdf_path),
        "page_count": len(documents),
        "chunk_count": len(chunks),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_index_meta(meta)
    return meta


def load_vector_store(persist_dir: Path = PERSIST_DIRECTORY) -> Chroma:
    """直接加载已经持久化到本地的 Chroma 向量库。"""
    return Chroma(
        persist_directory=str(persist_dir),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )


def create_or_load_retriever(
    pdf_path: Path = PDF_PATH,
    persist_dir: Path = PERSIST_DIRECTORY,
):
    """优先加载本地已有向量库；如不存在，则自动从 PDF 重建。"""
    if is_vector_store_ready(persist_dir):
        vector_store = load_vector_store(persist_dir)
        meta = load_index_meta()
        init_info = {
            "status": "loaded",
            "message": "检测到本地已有 Chroma 向量库，已直接加载。",
            "pdf_path": meta.get("pdf_path", str(pdf_path)),
            "page_count": meta.get("page_count"),
            "chunk_count": meta.get("chunk_count"),
            "updated_at": meta.get("updated_at"),
        }
    else:
        meta = build_vector_store(pdf_path, persist_dir)
        vector_store = load_vector_store(persist_dir)
        init_info = meta

    retriever = vector_store.as_retriever(
        search_kwargs={"k": CANDIDATE_TOP_K},
    )
    return retriever, init_info
