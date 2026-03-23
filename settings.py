"""
项目级配置。

这个文件只放常量，方便后续统一修改路径和参数。
"""

from pathlib import Path


# 学生手册 PDF 的固定路径。
# 后续如果你把 PDF 移到项目目录，只改这里即可。
PDF_PATH = Path(r"C:\Users\皓\Downloads\学生手册（教育管理篇）2025版.pdf")

# 向量库和索引元数据的本地存放位置。
VECTOR_DB_DIR = Path("chroma_db")
INDEX_META_PATH = Path("manual_index_meta.json")
SPLITTER_VERSION = "article_chunk_v2"

# 模型与检索相关配置。
MODEL_NAME = "glm-4"
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
CANDIDATE_TOP_K = 8
RETRIEVAL_TOP_K = 3
SCORE_THRESHOLD = 0.3

# PDF 切片参数。
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# 评测和异步问答时的最大并发数。
MAX_CONCURRENCY = 3
