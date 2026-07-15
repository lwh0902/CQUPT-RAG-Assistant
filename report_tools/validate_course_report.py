from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "203214445罗炜皓_基于RAG与Function Calling的校园制度智能问答系统设计与实现.docx"
REFS = json.loads((ROOT / "report_assets/evidence/references.json").read_text(encoding="utf-8"))
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}


def main() -> int:
    errors: list[str] = []
    if not DOCX.exists():
        print(f"FAIL: missing {DOCX}")
        return 1
    with zipfile.ZipFile(DOCX) as archive:
        document_xml = archive.read("word/document.xml")
        styles_xml = archive.read("word/styles.xml")
        rels_xml = archive.read("word/_rels/document.xml.rels")
        text = "".join(ET.fromstring(document_xml).itertext())
        styles_text = styles_xml.decode("utf-8", errors="ignore")
        rels_root = ET.fromstring(rels_xml)
        hyperlink_targets = {
            node.attrib.get("Target", "")
            for node in rels_root
            if node.attrib.get("Type", "").endswith("/hyperlink")
        }
        image_files = [name for name in archive.namelist() if name.startswith("word/media/")]

    required_text = [
        "203214445", "罗炜皓", "13902301", "计算机科学与技术学院", "软件工程", "赵志强",
        "2026 年 6 月 28 日", "1 项目背景及研究意义", "2 相关前沿技术", "3 系统需求分析",
        "4 系统总体设计", "5 核心功能设计与实现", "6 系统功能展示", "7 实验设计与结果分析",
        "8 项目创新点、不足与改进方向", "9 总结", "参考文献", "附录 A 现场演示建议",
        "66.7%", "41.7%", "329 个文本块",
    ]
    for item in required_text:
        if item not in text:
            errors.append(f"missing required text: {item}")

    if len(image_files) < 7:
        errors.append(f"expected at least 7 embedded images, found {len(image_files)}")
    if "宋体" not in styles_text:
        errors.append("宋体 is not declared in styles.xml")
    if "Times New Roman" not in styles_text:
        errors.append("Times New Roman is not declared in styles.xml")
    for ref in REFS:
        if ref["url"] not in hyperlink_targets:
            errors.append(f"missing hyperlink: {ref['url']}")
    forbidden = ["TBD", "TODO", "待补充", "PLACEHOLDER", "turn0search", "cite"]
    for token in forbidden:
        if token in text:
            errors.append(f"forbidden placeholder/citation token: {token}")

    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS")
    print(f"document: {DOCX}")
    print(f"embedded images: {len(image_files)}")
    print(f"verified reference hyperlinks: {len(REFS)}")
    print(f"text characters: {len(text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

