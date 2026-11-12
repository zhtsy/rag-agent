# -*- coding: utf-8 -*-
"""多格式加载：txt / md / pdf / docx / 图片 → 统一的 Doc 列表。

设计上的三个决定
────────────────
1. **重依赖全部延迟导入**。`import ingest.loaders` 只依赖标准库，pypdf / PIL /
   python-docx 都在函数里 import。理由很实在：`rag_agent` 一 import 就要加载
   1.2GB 的向量+重排模型，CI 里根本跑不了；这一层必须能在纯 pytest 环境下
   被 import 和单测。

2. **绝不静默产出空文档**。这是原来 `PyPDFLoader` 分支最坑的地方：扫描件
   抽出 0 字，不报错，最后表现为"知识库里有这个文件，但问它永远答不出来"。
   现在空页会带 `metadata["empty"]=True` 和 `needs_ocr`，由建库脚本汇总告警。

3. **OCR 与图像理解走注入**，不写死实现。`ocr=` 收一个 `PIL.Image -> str`
   的函数，`caption=` 同理（给 VLM 用）。缺 tesseract 时可以只跑 caption，
   或者在测试里塞一个假函数，不必为了测一行逻辑去装系统级 OCR。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

# 低于这个字数就当"没有文字层"。20 是实测出来的：正常一页正文 500 字以上，
# 页码/水印残留也就 5-10 字，中间没有重叠。
MIN_PDF_TEXT = 20

TEXT_EXTS = {".txt", ".md", ".markdown"}
PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DOCX_EXTS = {".docx"}
SUPPORTED_EXTS = TEXT_EXTS | PDF_EXTS | IMAGE_EXTS | DOCX_EXTS


class LoaderUnavailable(RuntimeError):
    """缺依赖或缺系统工具。调用方可以选择跳过并计数，而不是整体失败。"""


@dataclass
class Doc:
    """一个逻辑页/一段文本。刻意不继承 langchain 的 Document，保持零依赖。"""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # 空正文即"无内容"，便于 `if doc:`
        return bool(self.text.strip())


# ========== OCR ==========

def ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:
        return False
    return True


def _default_ocr(img: Any) -> str:
    """默认 OCR：pytesseract + 中英双语。没装就抛，由调用方决定降级。"""
    try:
        import pytesseract
    except Exception as exc:  # pragma: no cover - 环境相关
        raise LoaderUnavailable("未安装 pytesseract，无法对扫描件/图片做 OCR") from exc
    return pytesseract.image_to_string(img, lang="chi_sim+eng")


# ========== 各格式 ==========

def _load_text(path: str) -> List[Doc]:
    raw = None
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            with open(path, "r", encoding=enc) as f:
                raw = f.read()
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raise LoaderUnavailable(f"{path} 无法按 utf-8/gb18030 解码")
    return [Doc(raw, {"kind": "text"})]


def _pdf_page_images(path: str) -> Iterator[tuple[int, Any]]:
    """逐页取出内嵌位图，不依赖 poppler（pdf2image 需要系统装 poppler）。"""
    from pypdf import PdfReader
    from PIL import Image
    import io

    reader = PdfReader(path)
    for i, page in enumerate(reader.pages, start=1):
        try:
            images = list(page.images)
        except Exception:
            images = []
        for im in images:
            try:
                yield i, Image.open(io.BytesIO(im.data))
            except Exception:
                continue


def _pdf_tables_md(path: str) -> Dict[int, List[str]]:
    """用 pdfplumber 抽表格，转成 markdown。没有 pdfplumber 就返回空。

    为什么必须单独抽：pypdf 拿到的是"文字位置流"，表格会被拼成一行
    `菜系口味特点代表菜川菜麻辣鲜香…`——表头和数据分不开，chunk 里
    那张表就等于不存在。pdfplumber 能识别单元格边界。
    """
    try:
        import pdfplumber
    except Exception:
        return {}

    out: Dict[int, List[str]] = {}
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    continue
                mds: List[str] = []
                for rows in tables:
                    cleaned = [
                        [("" if c is None else str(c).replace("\n", " ").strip()) for c in row]
                        for row in rows or []
                    ]
                    cleaned = [r for r in cleaned if any(r)]
                    if len(cleaned) < 2:
                        continue
                    header, body = cleaned[0], cleaned[1:]
                    md = ["| " + " | ".join(header) + " |",
                          "| " + " | ".join(["---"] * len(header)) + " |"]
                    md += ["| " + " | ".join(r) + " |" for r in body]
                    mds.append("\n".join(md))
                if mds:
                    out[i] = mds
    except Exception:
        return out
    return out


def _drop_flattened_cells(text: str, table_mds: Sequence[str]) -> str:
    """把 pypdf 那行"表头数据糊在一起"的文字去掉，避免同一张表进库两遍。

    **按整行拼起来删，不要按单元格删**。第一版是按单元格 replace 的，
    结果正文标题里的「菜系」「代表菜」也被各删一次——
    "主要菜系对照表" 变成 "主要对照表"。整行拼接（`菜系口味特点代表菜`）
    只在表格被压平时才出现，误删概率极低；拼不上就说明 pypdf 没压平，
    那就不删，宁可重复也不破坏正文。
    """
    for md in table_mds:
        rows = [ln for ln in md.splitlines() if not ln.startswith("| ---")]
        for line in rows:
            joined = "".join(c.strip() for c in line.strip("|").split("|"))
            if len(joined) >= 6 and joined in text:
                text = text.replace(joined, "", 1)
    return text


def _load_pdf(
    path: str,
    ocr: Optional[Callable[[Any], str]],
    caption: Optional[Callable[[Any], str]],
) -> List[Doc]:
    from pypdf import PdfReader

    from .clean import clean_pages

    reader = PdfReader(path)
    raw_pages = [(page.extract_text() or "") for page in reader.pages]
    pages = clean_pages(raw_pages)
    tables = _pdf_tables_md(path)

    docs: List[Doc] = []
    for i, (raw, text) in enumerate(zip(raw_pages, pages), start=1):
        meta: Dict[str, Any] = {"kind": "pdf", "page": i, "empty": False}
        if i in tables:
            text = _drop_flattened_cells(text, tables[i])
            text = (text + "\n\n" + "\n\n".join(tables[i])).strip()
            meta["tables"] = len(tables[i])
        if len(text.strip()) < MIN_PDF_TEXT:
            # 文字层几乎没有 → 这一页是扫描件/整页图片
            meta["needs_ocr"] = True
            recovered = _recover_from_images(path, page_no=i, ocr=ocr, caption=caption)
            if recovered:
                text = recovered
                meta["ocr_used"] = True
                meta["empty"] = False
            else:
                meta["empty"] = True
        meta["raw_chars"] = len(raw or "")
        docs.append(Doc(text, meta))
    return docs


def _recover_from_images(
    path: str,
    page_no: int,
    ocr: Optional[Callable[[Any], str]],
    caption: Optional[Callable[[Any], str]],
) -> str:
    """从某一页的内嵌图片里抢救文字：先 caption（懂图表），再 OCR（认字）。"""
    parts: List[str] = []
    for pno, img in _pdf_page_images(path):
        if pno != page_no:
            continue
        for fn in (caption, ocr):
            if fn is None:
                continue
            try:
                out = (fn(img) or "").strip()
            except LoaderUnavailable:
                raise
            except Exception:
                continue
            if out:
                parts.append(out)
                break
    return "\n".join(parts).strip()


def _load_image(
    path: str,
    ocr: Optional[Callable[[Any], str]],
    caption: Optional[Callable[[Any], str]],
) -> List[Doc]:
    from PIL import Image

    img = Image.open(path)
    meta: Dict[str, Any] = {
        "kind": "image",
        "width": img.width,
        "height": img.height,
        "empty": False,
    }
    text = ""
    for name, fn in (("caption", caption), ("ocr", ocr)):
        if fn is None:
            continue
        try:
            text = (fn(img) or "").strip()
        except LoaderUnavailable:
            raise
        except Exception:
            text = ""
        if text:
            meta[f"{name}_used"] = True
            break
    if not text:
        meta["empty"] = True
        meta["needs_ocr"] = True
    return [Doc(text, meta)]


def _load_docx(path: str) -> List[Doc]:
    try:
        import docx  # python-docx
    except Exception as exc:  # pragma: no cover
        raise LoaderUnavailable("未安装 python-docx，无法解析 .docx") from exc

    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    # 表格单独抽，按行拼成 "| a | b |"，便于下游识别成表格块
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append("| " + " | ".join(cells) + " |")
    return [Doc("\n".join(parts), {"kind": "docx"})]


# ========== 入口 ==========

def load_file(
    path: str,
    *,
    ocr: Optional[Callable[[Any], str]] = None,
    caption: Optional[Callable[[Any], str]] = None,
) -> List[Doc]:
    """按后缀分发。返回的每个 Doc 都带 source_file / source_path。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXTS:
        docs = _load_text(path)
    elif ext in PDF_EXTS:
        docs = _load_pdf(path, ocr, caption)
    elif ext in IMAGE_EXTS:
        docs = _load_image(path, ocr, caption)
    elif ext in DOCX_EXTS:
        docs = _load_docx(path)
    else:
        return []

    base = os.path.basename(path)
    for d in docs:
        d.metadata.setdefault("source_file", base)
        d.metadata.setdefault("source_path", os.path.abspath(path))
        d.metadata.setdefault("ext", ext)
    return docs


def load_dir(
    root: str,
    *,
    exts: Optional[Iterable[str]] = None,
    ocr: Optional[Callable[[Any], str]] = None,
    caption: Optional[Callable[[Any], str]] = None,
    skip_dirs: Sequence[str] = ("__pycache__", ".git", "node_modules"),
    on_error: Optional[Callable[[str, Exception], None]] = None,
) -> List[Doc]:
    """递归加载目录。单个文件失败不影响其它文件。"""
    allow = {e.lower() for e in exts} if exts else SUPPORTED_EXTS
    out: List[Doc] = []
    for cur, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() not in allow:
                continue
            path = os.path.join(cur, fn)
            try:
                out.extend(load_file(path, ocr=ocr, caption=caption))
            except Exception as exc:
                if on_error:
                    on_error(path, exc)
                else:
                    raise
    return out


def stats(docs: Sequence[Doc]) -> Dict[str, int]:
    """建库脚本用来告警的汇总：多少页是空的、多少页靠 OCR 救回来的。"""
    return {
        "pages": len(docs),
        "empty": sum(1 for d in docs if d.metadata.get("empty")),
        "ocr_used": sum(1 for d in docs if d.metadata.get("ocr_used")),
        "chars": sum(len(d.text) for d in docs),
    }
