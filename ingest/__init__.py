# -*- coding: utf-8 -*-
"""多格式入库：解析 → 清洗 → 切块。

对外只暴露三样东西：
    load_file / load_dir    解析单文件 / 整个目录
    chunk_docs              切块
    stats                   汇总（多少页是空的、多少块是表格）

刻意零第三方依赖（重依赖在函数里延迟导入），所以能在纯 pytest 环境下单测。
"""

from .chunker import chunk_doc, chunk_docs
from .clean import clean_pages, clean_text, strip_page_furniture, unmangle_lines
from .loaders import (
    DOCX_EXTS,
    IMAGE_EXTS,
    PDF_EXTS,
    SUPPORTED_EXTS,
    TEXT_EXTS,
    Doc,
    LoaderUnavailable,
    load_dir,
    load_file,
    ocr_available,
)

__all__ = [
    "Doc",
    "LoaderUnavailable",
    "load_file",
    "load_dir",
    "chunk_doc",
    "chunk_docs",
    "clean_text",
    "clean_pages",
    "strip_page_furniture",
    "unmangle_lines",
    "ocr_available",
    "TEXT_EXTS",
    "PDF_EXTS",
    "IMAGE_EXTS",
    "DOCX_EXTS",
    "SUPPORTED_EXTS",
]
