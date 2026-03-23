"""
PDF 读取与切片模块。

这里专门负责：
1. 用 PyMuPDF 读取学生手册 PDF
2. 生成带页码信息的 Document 列表
3. 把长文本切成适合检索的小片段
"""

from pathlib import Path
import re

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import CHUNK_OVERLAP, CHUNK_SIZE


def normalize_page_text(text: str) -> str:
    """
    清洗单页文本，尽量保留语义结构，同时去掉明显空白噪声。
    """
    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines).strip()


def load_pdf_documents(pdf_path: Path) -> list[Document]:
    """
    读取整本 PDF，并返回按页组织的 Document 列表。

    每一页都会保留页码 metadata，方便后续展示引用来源。
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到学生手册 PDF：{pdf_path}")

    documents: list[Document] = []
    pdf = fitz.open(pdf_path)

    try:
        for page_index, page in enumerate(pdf, start=1):
            text = normalize_page_text(page.get_text())
            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_index,
                        "source": pdf_path.name,
                    },
                )
            )
    finally:
        pdf.close()

    if not documents:
        raise ValueError("PDF 文本提取结果为空，请检查文件是否为可提取文本的 PDF。")

    return documents


def split_page_into_sections(text: str) -> list[str]:
    """
    先按制度文本的结构做一次“粗切”。

    这里优先识别：
    - 第X章 / 第X节 / 第X条
    - （一）（二）（三）这类条款序号

    这样可以尽量让一个 chunk 保持在同一条款语义范围内，
    减少把不同奖项标准混在一起的情况。
    """
    marked_text = text

    # 在章、节、条前插入分隔符，优先保留制度结构。
    marked_text = re.sub(r"(第[一二三四五六七八九十百零〇\d]+[章节条])", r"\n@@\1", marked_text)

    sections = []
    for part in marked_text.split("\n@@"):
        cleaned = part.strip()
        if cleaned:
            sections.append(cleaned)

    # 如果某个 section 过短，通常只是标题被单独切出来了。
    # 这里把过短标题和后一个正文合并，避免“只检索到标题”的情况。
    merged_sections = []
    for section in sections:
        if merged_sections and len(merged_sections[-1]) < 30:
            merged_sections[-1] = f"{merged_sections[-1]}\n{section}".strip()
        else:
            merged_sections.append(section)

    return merged_sections


def split_documents(documents: list[Document]) -> list[Document]:
    """
    把按页提取的文档切成更适合向量检索的小片段。
    """
    section_documents: list[Document] = []

    # 先按制度条款做结构化切分，再对超长段落做二次切片。
    for doc in documents:
        sections = split_page_into_sections(doc.page_content)
        for section_index, section_text in enumerate(sections, start=1):
            section_documents.append(
                Document(
                    page_content=section_text,
                    metadata={
                        **doc.metadata,
                        "section_index": section_index,
                    },
                )
            )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "！", "？", " ", ""],
    )
    return text_splitter.split_documents(section_documents)
