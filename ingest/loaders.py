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
import re
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


# 有文字层、抽出来却是字形编号（`/G21 /G22 /G23 ...`）—— 这类 PDF 用了缺少
# ToUnicode 映射的子集字体（实测《计量法》2018 版就是），pypdf 只能拿到 glyph 名。
# 它字数看着不少（3 页 1.1 万字），但没有一个字是正文。如果当成正常文档入库，
# 后果比空页更糟：空页会被 needs_ocr 标出来，乱码块却会**安静地占满 top-5**，
# 而且 rerank 会认为它"很像法律文本"给高分 —— 表现出来是"答得一本正经地错"。
_GLYPH_RE = re.compile(r"/G\d+")

# 图片 OCR 结果的汉字占比下限。低于它说明 tesseract 把中文认成了拉丁字母
# （实测 `AM am, (exo maa, mE.` 汉字占比 0），这种页宁可不入库。
# 定 0.35 而不是 0.5：扫描件里表格页有大量数字与符号，真实汉字占比也就 0.6 上下，
# 门槛太高会把表格页一起挡掉 —— 而表格页恰恰是图片语料唯一有价值的部分。
_MIN_CJK_RATIO = 0.35


def looks_like_glyph_garbage(text: str) -> bool:
    """文字层是不是"只有字形编号、没有实际字符"。

    两个条件同时成立才判乱码：
      · 出现至少 5 个 `/G数字` 形态的 token；
      · 中文字符占比低于 15%。
    只满足第一条的可能是正常文档里恰好提到，宁可不判。
    """
    if not text:
        return False
    if len(_GLYPH_RE.findall(text)) < 5:
        return False
    n_cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    return n_cjk < max(3, len(text) * 0.15)


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


# ---- OCR 结果缓存 ----
# OCR 是建库最慢的一环：单页约 6 s，10 份扫描件跑一遍就是 20 分钟。而同一张
# 图的字节不变，OCR 结果就不会变 —— 每次重跑都重算纯属浪费（调去重规则、改切块
# 参数时尤其难受）。所以按图片内容做键缓存到磁盘。
#
# 改预处理逻辑（放大倍数、psm、语言包、分栏）时**必须同步改 _OCR_VERSION**，
# 否则会读到旧结果，还以为是新参数的效果——这类"改了没反应"最难查。
_OCR_VERSION = "colsplit-psm4-2x-gray-v5"
_OCR_CACHE_DIR = os.getenv("RAG_OCR_CACHE_DIR") or os.path.join(
    os.path.expanduser("~"), ".cache", "rag_ocr"
)


def _ocr_cache_path(img: Any) -> str:
    import hashlib

    h = hashlib.sha1()
    h.update(_OCR_VERSION.encode("utf-8"))
    h.update(f"{img.width}x{img.height}".encode())
    h.update(img.convert("RGB").tobytes())
    return os.path.join(_OCR_CACHE_DIR, h.hexdigest() + ".txt")


def _detect_columns(img: Any):
    """找双栏版面，返回 `(分栏横坐标, 双栏起点纵坐标)`；单栏返回 None。都是原图像素。

    为什么非得按栏切
    ────────────────
    人大常委会公报是双栏排版。tesseract 不管 `--psm 4` 还是 `--psm 3`，
    都是**按行横扫整幅图**，于是左右栏被逐行缝在一起：

        第四条 商业银行以安全性、流动性、效益     ← 左栏被右栏截断
        第二条                                  ← 右栏内容插进来
        ”性为经营原则，实行自主经草，自担风险     ← "效益性"被拆到另一行

    "效益性"三个字在库里成了 `效益` + `性`，关键词检索必然落空，向量也被稀释。
    这不是识别准确率问题——每个字都认对了，是**阅读顺序**错了。
    切栏之后每栏内部是单栏文本，psm 4 就成立了。

    为什么还要返回纵坐标
    ────────────────────
    公文首页上方有通栏内容：文号、标题「中华人民共和国商业银行法」、通过日期。
    整页对半切会把标题劈成「中华人民共和国商」+「业银行法」，比串行更糟。
    所以先沿栏间空白带逐行扫描，找出"这一行在两栏之间没有字"的最低位置，
    以它为界：上面通栏、下面双栏。实测这页的分界正好落在目录上方。

    判定方法：把图缩到 600 宽统计每列墨迹数。**注意墨迹不是行高**——
    正常正文每列只有 5%~10% 的像素是黑的（字与字之间有大量留白），
    所以"中位 63 / 行高 842"是正常值，不能拿 15% 当"空"的门槛（第一版就错在这）。
    """
    w, h = img.size
    if w < 200 or h < 200:
        return None
    scale = 600.0 / w
    small = img.convert("L").resize((600, max(1, int(h * scale))))
    sw, sh = small.size
    px = list(small.getdata())

    counts = [0] * sw
    for y in range(sh):
        base = y * sw
        for x in range(sw):
            if px[base + x] < 160:
                counts[x] += 1

    lo, hi = int(sw * 0.35), int(sw * 0.65)
    band = counts[lo:hi]
    if not band:
        return None
    ordered = sorted(band)
    median = ordered[len(ordered) // 2]
    if median < 3:                    # 中间带几乎没字 → 整页是图，谈不上分栏
        return None

    blank_thr = max(2, int(median * 0.06))
    # 取"最长连续空白列"的中点，比取单列最小值稳（扫描有噪点，孤立的 0 不可信）
    best_len, best_start, run_start = 0, -1, None
    for i, c in enumerate(band):
        if c <= blank_thr:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                if i - run_start > best_len:
                    best_len, best_start = i - run_start, run_start
                run_start = None
    if run_start is not None and len(band) - run_start > best_len:
        best_len, best_start = len(band) - run_start, run_start
    if best_len < 2:                  # 没有成气候的空白带 → 单栏
        return None

    gut = lo + best_start + best_len // 2
    left_ink = sum(counts[:gut])
    right_ink = sum(counts[gut + 1:])
    total = left_ink + right_ink
    if total == 0 or left_ink / total < 0.15 or right_ink / total < 0.15:
        return None                    # 一侧几乎没字，不是对等双栏

    # 逐行看栏间带：没有字 = 这一行属于双栏区。
    # 不能"从最后一行往上走、遇到有字就停"——页脚「—— 913 ——」正好压在栏间带上，
    # 第一步就会被它打断，判成"双栏区高度为 0"，分栏直接不生效（实测踩过）。
    # 改成"从某行往下，90% 以上的行都是双栏"的最低那一行：页脚、正文里偶发的
    # 通栏小标题都只是个别行，不会把整体比例拉下去。
    #
    # 扫描必须用**空白带的精确边界**，不能再加窗口：第一版按 gut±7 取窗口，
    # 窗口压到了邻栏字边，于是每一行都"不空"，双栏区高度算成 0，检测整体失效。
    run_lo, run_len = lo + best_start, best_len
    two = []
    for y in range(sh):
        base = y * sw
        blank = True
        for x in range(run_lo, run_lo + run_len):
            if px[base + x] < 160:
                blank = False
                break
        two.append(blank)

    suffix = [0] * (sh + 1)
    for y in range(sh - 1, -1, -1):
        suffix[y] = suffix[y + 1] + (1 if two[y] else 0)

    y_start = sh
    for y in range(sh):
        if suffix[y] / float(sh - y) >= 0.90:
            y_start = y
            break

    # 双栏区至少要占到下半页的三成，否则判定不可信（可能是表格或落款）
    if (sh - y_start) / sh < 0.30:
        return None
    if y_start < 3:                    # 顶端就是双栏，没有通栏内容
        y_start = 0

    return max(1, int(gut / scale)), int(y_start / scale)


# OCR 统计：建库脚本要报"多少页是双栏切开的"，否则分栏没生效也看不出来。
_OCR_STATS = {"pages": 0, "two_column": 0, "full_width_top": 0, "fallback_split": 0}


def interleave_ratio(text: str) -> float:
    """长行里"行中间有空白"的比例。串行了就是双栏页，没串行就是普通文本。"""
    lines = [ln.strip() for ln in (text or "").split("\n") if len(ln.strip()) >= 20]
    if not lines:
        return 0.0
    return sum(1 for ln in lines if _INTERLEAVE_RE.search(ln)) / float(len(lines))


def ocr_stats() -> Dict[str, int]:
    return dict(_OCR_STATS)


def _ocr_text(im: Any) -> str:
    import pytesseract

    return pytesseract.image_to_string(im, lang="chi_sim+eng", config="--psm 4")


def _default_ocr(img: Any) -> str:
    """默认 OCR：pytesseract + 中英双语。没装就抛，由调用方决定降级。

    三个预处理决定，都是拿真实扫描件（2006 年人大常委会公报）调出来的：
      · **灰度 + 2 倍放大**：原扫描 2480×3484，字已经够大，但 tesseract 在
        双栏公报上的分行不可靠，放大后「第五条」这类被印章切到的条号才认得回来
        （实测单页识别字数 1270 → 1572，条号从 11 个变 12 个）。
      · **`--psm 4`**（按列假设一段文本）比默认的 psm 3（全自动分页）稳：
        公文是双栏，psm 3 会自己去猜版面，猜错就把左右栏串行。
      · **检测到双栏就按栏切开分别识别**（2026-11-12 加）。psm 4 解决的是
        "怎么按行切"，解决不了"整幅图里有两栏"——它照样横扫，把两栏缝成一行。
        切栏后左栏读完再读右栏，才是人眼的顺序。判据见 `_column_gutter`。
    代价是像素翻 4 倍、单页从约 3 s 涨到约 6 s（双栏页约 12 s，要识别两栏）。
    扫描件建库是一次性开销，值。

    结果按图片内容缓存到 `~/.cache/rag_ocr/`（可用 RAG_OCR_CACHE_DIR 改）。
    """
    try:
        import pytesseract  # noqa: F401
        from PIL import Image
    except Exception as exc:  # pragma: no cover - 环境相关
        raise LoaderUnavailable("未安装 pytesseract，无法对扫描件/图片做 OCR") from exc

    cache_path = ""
    try:
        cache_path = _ocr_cache_path(img)
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        pass                     # 没缓存 / 目录不可写 → 照常 OCR
    except Exception:
        cache_path = ""           # 图片对象异常时不要因为缓存把整页搞失败

    im = img.convert("L")
    if im.width < 3000:
        im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)

    _OCR_STATS["pages"] += 1
    found = _detect_columns(im)
    if found is None:
        text = _ocr_text(im)
        # 兜底：目录页/栏宽不齐的页面，全局投影找不到干净空白带，检测会返回 None，
        # 但它们**照样是双栏**。所以整页识别完再验一次：真串行了就按中线切开重识，
        # 谁的交错行少用谁。宁可多花一次 OCR，也不要入库一份字序错乱的正文。
        if interleave_ratio(text) >= 0.15:
            gut = im.width // 2
            pad = max(10, int(im.width * 0.01))
            alt = (_ocr_text(im.crop((0, 0, min(im.width, gut + pad), im.height))).strip()
                   + "\n"
                   + _ocr_text(im.crop((max(0, gut - pad), 0, im.width, im.height))).strip()).strip()
            if alt and interleave_ratio(alt) < interleave_ratio(text):
                text = alt
                _OCR_STATS["two_column"] += 1
                _OCR_STATS["fallback_split"] += 1
    else:
        _OCR_STATS["two_column"] += 1
        gut, y0 = found
        # 两侧各留 1% 重叠：栏间空白可能压到字边，宁可重复几个字也别切掉
        pad = max(10, int(im.width * 0.01))
        parts: List[str] = []
        if y0 > 0:                     # 通栏部分：文号、标题、通过日期
            parts.append(_ocr_text(im.crop((0, 0, im.width, y0))).strip())
        parts.append(_ocr_text(im.crop((0, y0, min(im.width, gut + pad), im.height))).strip())
        parts.append(_ocr_text(im.crop((max(0, gut - pad), y0, im.width, im.height))).strip())
        if y0 > 0:
            _OCR_STATS["full_width_top"] += 1
        text = "\n".join(p for p in parts if p).strip()

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass                  # 缓存写失败不能影响主流程
    return text


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


def _cluster_by_x(chars: Sequence[Dict[str, Any]], tol: float = 6.0) -> List[List[Any]]:
    """把字符按横坐标聚成"竖列"。返回 [[x, [char, ...]], ...]，按 x 升序。"""
    cols: List[List[Any]] = []
    for c in sorted(chars, key=lambda z: float(z["x0"])):
        x = float(c["x0"])
        if cols and abs(x - cols[-1][0]) <= tol:
            cols[-1][1].append(c)
        else:
            cols.append([x, [c]])
    return cols


# 竖排单元格的判定：比它宽更"窄"，比它高更"高"。
# 实测《车船税法》附表第一列是 42pt 宽 × 147pt 高，装 20 个字。
_VERT_MAX_W = 60.0
_VERT_MIN_H = 60.0


def _vertical_cell_text(chars: Sequence[Dict[str, Any]], box: Sequence[float]) -> Optional[str]:
    """还原"竖排（旋转 90°）单元格"的文字顺序。不是竖排返回 None。

    为什么需要
    ──────────
    车船税法附表第一列写的是「乘用车〔按发动机汽缸容量（排气量）分档〕」，
    但它是旋转 90° 排的：pdfplumber 的 `extract_text` 按内容流吐字，结果变成

        乘 容 用 量 车 ︵ ︹ 排 按 气 发 量 动 ︶ 机 分 汽 档 缸 ︺

    这不是乱码，两次"隔一个取一个"就能拆出两列竖排（**每列从上往下**）：
      左列（x≈19）：乘 用 车 〔 按 发 动 机 汽 缸
      右列（x≈33）：容 量 （ 排 气 量 ） 分 档 〕
    左列在前拼起来正好是「乘用车〔按发动机汽缸容量（排气量）分档〕」。

    所以规则是**列按 x 升序、列内按 top 升序**。三条约束缺一不可：
      · 窄（<60pt）且高（>60pt）—— 普通横排单元格不会这样；
      · 2~3 列、每列至少 3 字 —— 排除"每字一列"的正常横排（`税 目` 有两列，
        但每列只有 1 字，走默认路径反而是对的）；
      · 总字数 ≥ 6 —— 太短看不出形态，不冒险。
    """
    x0, top, x1, bottom = (float(v) for v in box)
    if (x1 - x0) > _VERT_MAX_W or (bottom - top) < _VERT_MIN_H:
        return None
    if len(chars) < 6:
        return None
    cols = _cluster_by_x(chars)
    if not (2 <= len(cols) <= 3):
        return None
    if any(len(c[1]) < 3 for c in cols):
        return None
    return "".join(
        "".join(ch["text"] for ch in sorted(c[1], key=lambda z: float(z["top"])))
        for c in cols
    )


def _chars_in(chars: Sequence[Dict[str, Any]], box: Sequence[float]) -> List[Dict[str, Any]]:
    """取中心落在 box 内的字符。用中心点而不是相交面积，避免吃到邻格的字。"""
    x0, top, x1, bottom = (float(v) for v in box)
    out = []
    for c in chars:
        cx = (float(c["x0"]) + float(c["x1"])) / 2.0
        cy = (float(c["top"]) + float(c["bottom"])) / 2.0
        if x0 - 1 <= cx <= x1 + 1 and top - 1 <= cy <= bottom + 1:
            out.append(c)
    return out


def _chars_to_text(chars: Sequence[Dict[str, Any]], line_tol: Optional[float] = None) -> str:
    """把一格内的字符拼成文本：按 top 分行，行内按 x 排序。

    不用 `crop.extract_text()`：它会把竖排的每列当成独立行（见
    `_vertical_cell_text`），而且相邻格的字符偶尔漏进来。

    **空白字符必须先剔掉再分行**。政府公报的表头「税　目」中间是两个全角空格，
    它们的 top 比汉字低 8.5pt（空格用不同基线排版），一旦参与行分组就被判成
    独立行，结果拼出「税 / 空格 / 目」三行；同理「1.2元至12元」会被拆成
    「元至元」和「1.212」—— 数字直接变成错的。空格改由"字距"在拉丁字符之间
    补回来（中文本不需要空格）。

    行容差**按字号自适应**（1.2 倍中位字高）：同一个单元格里数字和汉字的基线
    能差 8.5pt（字号 9.7），但真正的下一行行距是 14~26pt。写死 4pt 会把
    「1.2元至12元」拦腰断开，写死 12pt 又会把相邻两行粘成一行。
    """
    words = [c for c in chars if (c.get("text") or "").strip()]
    if not words:
        return ""
    if line_tol is None:
        sizes = sorted(float(c.get("size") or 0) for c in words)
        median = sizes[len(sizes) // 2] if sizes else 0.0
        line_tol = max(4.0, 1.2 * median) if median > 0 else 4.0
    words.sort(key=lambda z: (float(z["top"]), float(z["x0"])))
    lines: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_top = None
    for c in words:
        t = float(c["top"])
        if cur_top is None or abs(t - cur_top) <= line_tol:
            cur.append(c)
            cur_top = t if cur_top is None else cur_top
        else:
            lines.append(cur)
            cur, cur_top = [c], t
    if cur:
        lines.append(cur)

    out = []
    for ln in lines:
        ln.sort(key=lambda z: float(z["x0"]))
        s = ""
        prev = None
        for c in ln:
            if prev is not None:
                gap = float(c["x0"]) - float(prev["x1"])
                a = (prev.get("text") or " ")[-1:]
                b = (c.get("text") or " ")[:1]
                if gap > 1.5 and re.match(r"[A-Za-z0-9]", a) and re.match(r"[A-Za-z0-9]", b):
                    s += " "
            s += c["text"]
            prev = c
        if s.strip():
            out.append(s.strip())
    return " ".join(out)


# 页眉/页码的长相：`2025·6`、`第 3 页`、`全国人民代表大会常务委员会公报2019·3`。
# 表格上沿紧跟着页眉时，取"最后一非空行"会把它当表名。
_RUNNING_HEAD_RE = re.compile(r"公报|第\s*\d+\s*[页期]|^\d{4}\s*[·．.]\s*\d+$|^20\d{2}$")


def _caption_above(page: Any, top: float, gap: float = 45.0) -> str:
    """表格正上方的表名。从下往上找第一行"不像页眉"的短文本。

    为什么必须带上：表本身没有"我叫什么"的信息，整张税率表入库后，问
    「车船税的税目税额表」这个词在块里一次都不出现，纯向量检索召不回来。

    为什么要跳过页眉：《环境保护税法》的表紧贴版面顶部，上沿 45pt 内最后一行
    是 `2025·6`（公报期号）。不排除的话 8 张表的表名全变成 `2025·6`，
    既没用又会让这 8 块在检索时挤在同一个假主题上。
    """
    y0 = max(0.0, float(top) - gap)
    try:
        text = page.crop((0, y0, page.width, float(top))).extract_text() or ""
    except Exception:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if len(ln) > 40 or _RUNNING_HEAD_RE.search(ln):
            continue
        return ln
    return ""


def _pdf_tables(path: str) -> Dict[int, Dict[str, Any]]:
    """抽出每页的表格（markdown）与"表格之外"的正文。

    `_pdf_tables_md` 只给表格 md，正文还是 pypdf 那份压平文本，两张都在库里。
    早先靠"整行拼接后再删"（`_drop_flattened_cells`）去重，但表格一旦被排成
    多行就匹配不上，压平的那份会原样留在块里。这里改成正面做法：用
    `outside_bbox` 把表格框外的正文取出来，调用方直接用它替代 pypdf 文本，
    一份内容只进一次。

    返回 {页号: {"tables": [md, ...], "body": 表格外正文}}
    """
    try:
        import pdfplumber
    except Exception:
        return {}

    from .clean import collapse_cjk_spaces, normalize_fullwidth_alnum

    out: Dict[int, Dict[str, Any]] = {}
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    found = page.find_tables() or []
                except Exception:
                    continue
                if not found:
                    continue
                mds: List[str] = []
                page_chars = page.chars
                for t in found:
                    try:
                        trows = t.rows
                    except Exception:
                        continue
                    # 逐格取字：None = 这一格被合并掉了；"" = 真的没写字。
                    # 两者必须分开：前者要按几何向下填充（「每辆」跨 8 行），
                    # 后者不能填（挂车没有子类，填成「货车」就是事实错误 ——
                    # 实测盲填曾产出「摩托车 | 轮式专用机械车」这种错行）。
                    grid: List[List[Optional[str]]] = []
                    for row in trows:
                        line: List[Optional[str]] = []
                        for cell in row.cells:
                            if cell is None:
                                line.append(None)
                                continue
                            chars = _chars_in(page_chars, cell)
                            vert = _vertical_cell_text(chars, cell)
                            line.append(vert if vert is not None else _chars_to_text(chars))
                        grid.append(line)

                    for ri in range(1, len(grid)):
                        row_top = float(trows[ri].bbox[1])
                        for ci in range(len(grid[ri])):
                            if grid[ri][ci] is not None:
                                continue
                            for rj in range(ri - 1, -1, -1):
                                if ci >= len(trows[rj].cells):
                                    break
                                up = trows[rj].cells[ci]
                                # 上方那格的下沿越过本行上沿 = 合并单元格跨过来了
                                if up is not None and float(up[3]) > row_top + 1:
                                    grid[ri][ci] = grid[rj][ci]
                                    break
                            if grid[ri][ci] is None:
                                grid[ri][ci] = ""

                    rows: List[List[str]] = [
                        [re.sub(r"\s{2,}", " ", (c or "").replace("\u3000", " ")).strip()
                         for c in line] for line in grid
                    ]
                    rows = [r for r in rows if any(r)]
                    if len(rows) < 2:
                        continue
                    header, body = rows[0], rows[1:]
                    md = ["| " + " | ".join(header) + " |",
                          "| " + " | ".join(["---"] * len(header)) + " |"]
                    md += ["| " + " | ".join(r) + " |" for r in body]
                    cap = collapse_cjk_spaces(_caption_above(page, t.bbox[1]))
                    text = "\n".join(md)
                    if cap:
                        # 表名单独一行，既让检索能命中，又不破坏 markdown 表结构
                        text = f"{cap}\n{text}"
                    mds.append(normalize_fullwidth_alnum(text))
                if not mds:
                    continue

                # 表格框外的正文：逐个 outside_bbox 挖掉表格区域
                body_page = page
                for t in found:
                    try:
                        body_page = body_page.outside_bbox(t.bbox)
                    except Exception:
                        pass
                try:
                    body = body_page.extract_text() or ""
                except Exception:
                    body = ""
                out[i] = {"tables": mds, "body": body}
    except Exception:
        return out
    return out


def _pdf_tables_md(path: str) -> Dict[int, List[str]]:
    """兼容旧接口：只要表格 md。"""
    return {p: v["tables"] for p, v in _pdf_tables(path).items()}


# 双栏页被"逐行缝起来"的特征：同一行里两栏的碎片之间留着栏间空白（3 个以上空格）。
# 正常单栏正文不会在行中间出现这种空洞——中文排版里字间空格会被 normalize 掉。
_INTERLEAVE_RE = re.compile(r"\S[ \t\u3000]{3,}\S")


def looks_interleaved(text: str) -> bool:
    """这一页的文字层是不是"左右栏交错"的。

    判据：长行里出现"行中间有一片空白"的比例超过 15%。
    要求至少 8 个长行才判，短页（封面、签章页）样本不够，宁可不改。

    为什么这比"文种/来源"更可靠：同一批 PDF 里既有双栏公报也有单栏排版，
    按文件名判断一定会错；而交错是排版产物，行内空白是它的直接指纹。
    """
    if len([ln for ln in (text or "").split("\n") if len(ln.strip()) >= 20]) < 8:
        return False
    return interleave_ratio(text) >= 0.15


def _join_words(words: Sequence[Dict[str, Any]]) -> str:
    """把同一行的词拼回去。中文之间不加空格，英文/数字之间要加。"""
    out = ""
    for w in words:
        t = str(w.get("text") or "")
        if not t:
            continue
        if out and re.match(r"[A-Za-z0-9]", t[-1]) and re.match(r"[A-Za-z0-9]", t[0]):
            out += " "
        out += t
    return out


def _pdf_layout_text(path: str) -> Dict[int, str]:
    """按字符坐标重建每页的阅读顺序，返回 {页号: 文本}。失败返回空字典。

    为什么不能靠 pypdf 的 extract_text
    ──────────────────────────────────
    它按**内容流顺序**吐字，而不是按版面。公报 PDF 的内容流是
    「左栏第 1 行 → 右栏第 1 行 → 左栏第 2 行 …」，于是拿到的就是交错文本：

        第四条 商业银行以安全性、流动性、效益     ← 左栏被右栏截断
        第二条                                  ← 右栏内容插进来

    pypdf 只是在 y 变化处补换行，几何信息（每个词的 x0/x1/top）它并不给我们。
    pdfplumber 给得到，所以这里自己按「先分栏、栏内自上而下、行内自左而右」重排。

    通栏内容（文号、居中标题）单独处理：一个词的横向跨越了栏间空白（x0 < 栏线 < x1）
    就说明它是通栏的，必须先于两栏输出——否则标题会被劈成两半塞进栏首。
    """
    try:
        import pdfplumber
    except Exception:
        return {}

    out: Dict[int, str] = {}
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
                except Exception:
                    continue
                if len(words) < 20:
                    continue
                width = float(page.width or 0) or 1.0

                # 找栏间空白：把词的中心点投影到 200 个格子上，看中间 30% 有没有空段
                bins = [0] * 200
                for w in words:
                    cx = (float(w["x0"]) + float(w["x1"])) / 2.0
                    bins[min(199, max(0, int(cx / width * 200)))] += 1
                lo, hi = 70, 130
                best_len, best_start, run = 0, -1, None
                for b in range(lo, hi):
                    if bins[b] == 0:
                        if run is None:
                            run = b
                    else:
                        if run is not None:
                            if b - run > best_len:
                                best_len, best_start = b - run, run
                            run = None
                if run is not None and hi - run > best_len:
                    best_len, best_start = hi - run, run
                if best_len < 3:
                    continue                      # 找不到栏间空白 → 单栏，别动它
                gut = (best_start + best_len / 2.0) / 200.0 * width

                wide, col0, col1 = [], [], []
                for w in words:
                    x0, x1 = float(w["x0"]), float(w["x1"])
                    if x0 < gut < x1:
                        wide.append(w)            # 通栏：标题、文号
                    elif (x0 + x1) / 2.0 < gut:
                        col0.append(w)
                    else:
                        col1.append(w)
                if not col0 or not col1:
                    continue

                def to_lines(ws):
                    ws = sorted(ws, key=lambda w: (round(float(w["top"]) / 3.0), float(w["x0"])))
                    lines, cur, cur_top = [], [], None
                    for w in ws:
                        t = float(w["top"])
                        if cur_top is None or abs(t - cur_top) <= 3.0:
                            cur.append(w)
                            cur_top = t if cur_top is None else cur_top
                        else:
                            lines.append(_join_words(sorted(cur, key=lambda x: float(x["x0"]))))
                            cur, cur_top = [w], t
                    if cur:
                        lines.append(_join_words(sorted(cur, key=lambda x: float(x["x0"]))))
                    return [ln for ln in lines if ln.strip()]

                text = "\n".join(to_lines(wide) + to_lines(col0) + to_lines(col1))
                if text.strip():
                    out[i] = text
    except Exception:
        return out
    return out


def _load_pdf(
    path: str,
    ocr: Optional[Callable[[Any], str]],
    caption: Optional[Callable[[Any], str]],
) -> List[Doc]:
    from pypdf import PdfReader

    from .clean import clean_pages, clean_text, isolate_article_lines

    reader = PdfReader(path)
    raw_pages = [(page.extract_text() or "") for page in reader.pages]

    # 表格先抽。抽到的页要用"表格外正文 + markdown 表"替代 pypdf 的压平文本，
    # 所以它必须在选页、清洗之前就拿到。
    tbl = _pdf_tables(path)
    effective = list(raw_pages)
    for pno, v in tbl.items():
        if 1 <= pno <= len(effective) and v.get("body", "").strip():
            effective[pno - 1] = v["body"]

    # 双栏页的文字层字序是错的（详见 looks_interleaved）。只有确实中招的文档
    # 才去开 pdfplumber 重建版面——它是纯 Python，全量跑一遍要几分钟，不能无条件用。
    # **含表格的页不参与重排**：表格中间那列空白在几何上和"栏间带"一模一样，
    # 重排会把整张表从中间切开，左半张塞进左栏、右半张塞进右栏。
    layout: Dict[int, str] = {}
    if any(looks_interleaved(r) for i, r in enumerate(raw_pages, 1)
           if not (i in tbl and tbl[i].get("tables"))):
        layout = _pdf_layout_text(path)

    pages = clean_pages(effective)

    docs: List[Doc] = []
    for i, (raw, text) in enumerate(zip(raw_pages, pages), start=1):
        meta: Dict[str, Any] = {"kind": "pdf", "page": i, "empty": False}
        has_table = bool(i in tbl and tbl[i].get("tables"))
        if has_table:
            meta["tables"] = len(tbl[i]["tables"])
            meta["table_page"] = True
            text = (text.strip() + "\n\n" + "\n\n".join(tbl[i]["tables"])).strip()
        elif looks_interleaved(raw):
            fixed = layout.get(i) or ""
            if fixed and not looks_interleaved(fixed):
                text = clean_text(fixed)
                meta["relayout"] = True
        # 有文字层但全是字形编号 → 当成"这页没文字"处理，试 OCR 兜底。
        # 不能直接入库：它会让检索"看起来有证据"，实际一个字都不对。
        if looks_like_glyph_garbage(text):
            meta["garbage"] = True
            recovered = _recover_from_images(path, page_no=i, ocr=ocr, caption=caption)
            if recovered and not looks_like_glyph_garbage(recovered):
                text = _clean_ocr(recovered)
                meta["ocr_used"] = True
            else:
                text = ""
                meta["empty"] = True
                meta["needs_ocr"] = True
        if len(text.strip()) < MIN_PDF_TEXT:
            # 文字层几乎没有 → 这一页是扫描件/整页图片
            meta["needs_ocr"] = True
            recovered = _recover_from_images(path, page_no=i, ocr=ocr, caption=caption)
            if recovered:
                # OCR 原文必须过一遍 clean_text：硬换行不接回来，「第三十条」会
                # 被拆成「第三十条」+「　当事人…」两段，切块时条号对不上。
                # `drop_garbage_lines` 清掉"中文被认成拉丁字母"的整行噪声。
                text = _clean_ocr(recovered)
                meta["ocr_used"] = True
                meta["empty"] = False
            else:
                meta["empty"] = True
        meta["raw_chars"] = len(raw or "")
        # 条号必须落在段落边界上，切块器才认得出"这是哪一条"（md 语料天然如此，
        # 只有 PDF 需要这步，原因见 clean.isolate_article_lines）。放在最后一次
        # 赋值之后，OCR 救回来的文本也能享受同样的结构化。
        docs.append(Doc(isolate_article_lines(text), meta))
    return docs


def _clean_ocr(raw: str) -> str:
    """OCR 原文 → 干净正文。**过滤要跑两遍**。

    乱码是"接行之后才显形"的。实测《诉讼费用交纳办法》第 2 页，tesseract 把
    「（五）申请认定财产无主」认成两行：

        `C1) RA`                 ← 6 字符，不够长，第一遍放行
        `效力;           re`     ← 汉字占比 0.40，第一遍也放行
        ──────────────────────────  clean_text 接行之后
        `C1) RA效力;           re` ← 汉字占比 0.15，这时才该删

    所以顺序是：先删一遍明显的（省得接行时把噪声粘进正文），接行，再删一遍。
    """
    from .clean import clean_text, drop_garbage_lines

    text = clean_text(drop_garbage_lines(raw))
    return drop_garbage_lines(text)


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
    """单张图片：OCR（或看图说话）→ 清洗 → 条号断行。

    为什么图片也要走 `clean_text` / `isolate_article_lines`（2026-11-13）
    ────────────────────────────────────────────────────────────────
    原来这里是直接返回 OCR 原文，实测出两串东西：

        `AM am, (exo maa, mE.`          ← psm4 把横排小字认成了拉丁字母
        `BAM me IES Al FSM:`            ← 同上，后面还跟着 (—) RRS. RFRA

    第一串一个汉字都没有，等于往索引里灌噪声；第二串丢了条号，图片块就永远
    对不上 md/pdf 的同一条。PDF 那条路早就洗过（`clean_pages`），图片这条路
    漏了 —— 同一个解析器里两条路走不同标准，是这类 bug 的常见来源。

    清洗后仍不合格（汉字占比 < `_MIN_CJK_RATIO`）的页按空页处理：宁可不入库，
    也不能让 OCR 噪声当成证据被召回。
    """
    from PIL import Image

    from .clean import cjk_ratio, isolate_article_lines

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
            raw = fn(img) or ""
        except LoaderUnavailable:
            raise
        except Exception:
            raw = ""
        if raw.strip():
            meta[f"{name}_used"] = True
            text = isolate_article_lines(_clean_ocr(raw))
            break
    if text and cjk_ratio(text) < _MIN_CJK_RATIO:
        # OCR 出来的"英文乱码"是中文被认错，不是真的外文：用汉字占比兜住。
        meta["garbage"] = True
        text = ""

    # 表格结构化：整页 OCR 把表摊成一句话（表头、单位、数值糊在一起），
    # 检索对不上、模型也看不出行列。这里用框线几何逐格还原（见 ingest/imgtable）。
    # 放在 OCR 之后是因为两者互不依赖：正文还得靠整页 OCR，表格靠网格。
    tables: List[str] = []
    if ocr is not None:
        try:
            from .imgtable import extract_tables
            tables = extract_tables(img)
        except Exception:
            tables = []
    if tables:
        meta["tables"] = len(tables)
        body = "\n\n".join(tables)
        # 正文被判成乱码时只留表格：表格是数字和短词，汉字占比天然偏低，
        # 不能拿页级门槛去卡它。
        text = (text + "\n\n" + body).strip() if text else body
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
        # 「有文字层但是字形编号」的页数。它和 empty 不是一回事：empty 是抽不出字，
        # garbage 是抽出了一堆假的字 —— 后者必须单独报出来，否则会被当成正常内容。
        "garbage": sum(1 for d in docs if d.metadata.get("garbage")),
        # 双栏页按坐标重排过的页数。它不该等于全部 PDF 页——只命中真交错的那些。
        "relayout": sum(1 for d in docs if d.metadata.get("relayout")),
        "chars": sum(len(d.text) for d in docs),
    }
