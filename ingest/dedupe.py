# -*- coding: utf-8 -*-
"""同一条文多格式并存时的去重。

为什么需要
──────────
同一部法可能同时以 `.md`（人工整理）和 `.pdf`（官方原件）进库。两条内容几乎
一样的块同时存在，后果不是"多一点冗余"这么轻：

  · top-5 名额被同义块吃掉 —— 五条证据里有三条在说同一件事，另外两个真正需要
    的条文被挤出去，答案就开始漏；
  · rerank 分数被抬高 —— 同一个句子出现两次，模型更"确信"，但那份确信是假的；
  · 溯源变得二义 —— 引用时说不清该指 md 还是 pdf，用户点开发现两份一样的。

所以按「法名 + 条号」判重，同一条只留一份。

为什么 md 优先
──────────────
md 那套已经跑出 0.894 的基线，条号边界干净、法名带标题路径；PDF 是这一轮新
接进来的，抽取质量受版式影响。**保留已验证的那份**，把 PDF 当作"补充 pdf 独有
内容"的角色：md 里没有的条（附表、修正说明、md 漏收的条文）照样进来。

判重键怎么取
────────────
有 `article` 元数据的块（按条切出来的），键是 `(法名, 条号)`，跨格式能对上。
没有条号的块（前言、章节导语、目录）不能一律当"重复"删掉 —— md 的前言和 PDF
的前言并不相同。这类退化成 `(法名, 正文前 60 字)`，只有正文确实一样才算重复。

**一个组里只能留一块，是错的**（2026-11-12 修）
──────────────────────────────────────────────
第一版是"每组排序后取第一名"，看着很自然，实际把同一格式**自己**也合并了：
一个长条文被切块器切成两段，两段的 `article` 都是「第三十六条」，于是后一段
被当重复丢掉。这个损失在建库日志里完全看不出来（只会看到"去掉 N 块"），
但实测把 47 题的语料削掉了一千多块：

    刑法.md      636 → 515    公司法.md   305 → 268    商标法.md  122 → 89

后果是定义式问题从 0.75 掉到 0.5 —— 定义往往就写在长条文的**后半段**，
被删掉的那块恰好是答案所在。所以现在的规则是：

  · 组里有 md/txt → 留**全部** md/txt 块（同格式多块是"一条被切成几段"，不是重复），
    丢掉这个组里所有 PDF 块（这一条 md 已经有了，PDF 版本不增加信息）；
  · 组里只有 PDF → 全部保留。它是这条的唯一来源，再合并就是删内容。
"""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Sequence, Tuple

# PDF 文件名尾部是 `_20201017_ff808081752b7d43`（日期 + 对象 id）。
_DATE_HASH_RE = re.compile(r"_\d{8}_[0-9a-f]{6,}$")

# 图片按页导出时尾部是 `_p012` / `_page12`。不剥掉的话 `专利法_p012.png` 会被
# 当成另一部叫「专利法_p012」的法，条号去重与 md 让位全部失效——图片块于是
# 变成"同一部法的第三份抄本"，正是把召回压到 0.861 的那个坑。
_PAGE_SUFFIX_RE = re.compile(r"[_\-](?:p|pg|page)\d{1,4}$", re.IGNORECASE)

# 这几个后缀被认为"已验证、优先保留"
DEFAULT_PREFER = (".md", ".markdown", ".txt")

# 无条号块的正文指纹长度。太短会把不同段落撞在一起，太长则漏掉真正的重复。
_FINGERPRINT_CHARS = 60


def law_name(source_file: str) -> str:
    """从文件名推法名：`中华人民共和国专利法_20201017_ab12.pdf` → `专利法`。

    去掉「中华人民共和国」前缀是因为 md 那边就叫 `专利法.md`，不去掉永远对不齐。
    图片按页导出（`专利法_p012.png`）同样归到 `专利法`，否则图片进库就绕过了
    全部去重规则。
    """
    stem = os.path.splitext(os.path.basename(source_file or ""))[0]
    stem = _DATE_HASH_RE.sub("", stem)
    stem = _PAGE_SUFFIX_RE.sub("", stem)
    return stem.replace("中华人民共和国", "").strip()


def _ext(meta: Dict) -> str:
    ext = str(meta.get("ext") or "").lower()
    if ext:
        return ext
    return os.path.splitext(str(meta.get("source_file") or ""))[1].lower()


def _fingerprint(text: str) -> str:
    return re.sub(r"\s+", "", text or "")[:_FINGERPRINT_CHARS]


def dedupe_key(chunk) -> Tuple[str, str]:
    """(法名, 条号) 或 (法名, 正文指纹)。

    条号优先：两个格式的同一法条，切块边界可能不同，但「第三十六条」一定一样。
    """
    meta = chunk.metadata or {}
    law = law_name(str(meta.get("source_file") or ""))
    article = str(meta.get("article") or "").strip()
    if article:
        return law, article
    return law, "text:" + _fingerprint(chunk.text)


def _rank(chunk, prefer: Sequence[str]) -> Tuple:
    """组内排序：偏好后缀 > 正文更长 > 文件名（保证结果可复现）。

    "正文更长"排第二是因为同名条号下，被截断的那份通常更短 —— 比如 md 把一条
    拆成两段，第一段单独成块；这时 PDF 的完整版反而更该留。但**只在同后缀内**
    才允许覆盖面优先（prefer 在前，长文只能战胜同后缀的短文）。
    """
    meta = chunk.metadata or {}
    ext = _ext(meta)
    tier = 0 if ext in prefer else 1
    return (tier, -len(chunk.text or ""), str(meta.get("source_file") or ""))


def dedupe_by_article(
    chunks: Sequence,
    prefer: Sequence[str] = DEFAULT_PREFER,
) -> Tuple[List, Dict[str, int]]:
    """按「法名 + 条号」去重，返回 (保留的块, 统计)。

    `prefer` 里的后缀优先保留，其余格式只补它没有的内容。**同格式内部的多个块
    不算重复**——那是一条长条文被切成了几段，合并会丢正文。
    """
    groups: Dict[Tuple[str, str], List] = {}
    for c in chunks:
        groups.setdefault(dedupe_key(c), []).append(c)

    kept: List = []
    dropped = 0
    kept_from_pdf = 0
    dup_groups = 0
    for group in groups.values():
        group.sort(key=lambda c: _rank(c, prefer))
        pref = [c for c in group if _ext(c.metadata or {}) in prefer]
        if pref:
            # 偏好格式全留；同组里的 PDF 块是同一内容的另一份，丢
            kept.extend(pref)
            rest = [c for c in group if _ext(c.metadata or {}) not in prefer]
            if rest:
                dup_groups += 1
                dropped += len(rest)
        else:
            # 没有偏好格式 → 这些块是这条的唯一来源，全留
            kept.extend(group)
            kept_from_pdf += len(group)

    stats = {
        "before": len(chunks),
        "after": len(kept),
        "dropped": dropped,
        "dup_groups": dup_groups,
        "kept_non_preferred": kept_from_pdf,
    }
    return kept, stats


def mark_shadowed_laws(
    chunks: Sequence,
    prefer: Sequence[str] = DEFAULT_PREFER,
) -> Tuple[List, Dict[str, int]]:
    """md 已经有的那部法，**PDF 版本保留但打上 shadow 标记**。

    为什么不删（2026-11-13 改）
    ──────────────────────────
    上一版是整部丢弃（`drop_shadowed_laws`）。那个决定混淆了两件事：
    **存储**和**召回**。用户要的是"检索时 md 优先"，不是"库里不该有 PDF"。

    而且一刀切丢弃是**不可逆的信息损失**：
      · 真重复（同法同条）已经在上一步 `dedupe_by_article` 里删掉了，走到这里
        还剩的 PDF 块，是 md 没覆盖的那部分 —— 不同版本的表述、附表页、封面题注；
      · md 那块万一被切坏、被 `_drop_noisy_image_chunks` 之类过滤掉，PDF 那份是
        唯一兜底；整部丢掉就把兜底一起丢了。

    改成标记之后，容量代价用 `supplement_stats` 量：留下的 PDF 块占全库多少。
    检索侧的优先级由 rag_agent 的 `RAG_SHADOW_PENALTY` 实现（同分时 md 胜）。
    """
    covered = set()
    for c in chunks:
        meta = c.metadata or {}
        if _ext(meta) in prefer:
            covered.add(law_name(str(meta.get("source_file") or "")))

    stats = {"before": len(chunks), "after": len(chunks), "shadowed": 0,
             "laws_covered": len(covered)}
    for c in chunks:
        meta = c.metadata
        if meta is None:
            continue
        ext = _ext(meta)
        if ext == ".pdf" and law_name(str(meta.get("source_file") or "")) in covered:
            meta["shadow"] = 1
            stats["shadowed"] += 1
        else:
            meta.setdefault("shadow", 0)
    return list(chunks), stats


def drop_shadowed_laws(
    chunks: Sequence,
    prefer: Sequence[str] = DEFAULT_PREFER,
) -> Tuple[List, Dict[str, int]]:
    """（旧行为，保留给对照/测试）md 已经有的那部法，PDF 版本整部丢掉。

    为什么按「法」而不是按「条」去重（2026-11-13）
    ────────────────────────────────────────────
    按条去重看起来更精细，但它依赖两边都切出了**同一个条号**。PDF 那边是 OCR
    出来的版面还原，条号经常认成 `第三百五十七条` 之外的形态，或者干脆没被
    `isolate_article_lines` 认成段首；键一对不上，去重就等于没做。实测：

        刑法        636 块 md  +  935 块 pdf   ← 几乎整部重复
        商标法      122 块 md  +   28 块 pdf
        个人所得税   36 块 md  +   24 块 pdf

    后果不是"多一点冗余"：47 题评测里召回从 **1.000 掉到 0.861**，
    L014 / D001 / D010 直接检索落空——top-5 名额被同一部法的另一份抄本占掉。

    所以改成法级让位：**md 有的法，PDF 一块都不进**；md 没有的（商业银行法、
    邮政法、国旗法…）才由 PDF 供给。

    让位的代价是"md 漏收的条文一起丢"，这条用 `supplement_stats` 量过：
    6,949 块被丢的 pdf 里独有条文 **0 块**，另有 95 块封面题注。所以这次让位
    没有丢内容。哪天语料变了（md 版本旧了、或换成节选本），建库日志的那张
   体检表会先变，再决定要不要改成按条。

    为什么要收窄成「只让 PDF」（2026-11-13 晚）
    ──────────────────────────────────────────
    原来是"除 md/txt 之外全让位"，图片也被挡在外面。但图片和 PDF 不是一回事：
      · PDF 是同一部法的**整部电子版**，和 md 是同一份内容的两个封装，必然重复；
      · 图片是**人工挑的少数几页**，恰恰常是文本层抽不出来的表格、印章、截图。
    把图片也整部让位，等于把唯一能承载表格的入口关掉。所以图片只走
    `dedupe_by_article`（同法同条/同指纹才丢），不做整部让位。
    """
    covered = set()
    for c in chunks:
        meta = c.metadata or {}
        if _ext(meta) in prefer:
            covered.add(law_name(str(meta.get("source_file") or "")))

    kept, dropped = [], 0
    for c in chunks:
        meta = c.metadata or {}
        ext = _ext(meta)
        if ext in prefer or ext != ".pdf":
            kept.append(c)
            continue
        if law_name(str(meta.get("source_file") or "")) in covered:
            dropped += 1
            continue
        kept.append(c)

    return kept, {"before": len(chunks), "after": len(kept), "dropped": dropped,
                  "laws_covered": len(covered)}


def _looks_like_table(text: str) -> bool:
    """没有 markdown 竖线时，认表格靠「多行都是多列数字」。

    一开始用的是整体数字密度（数字/字数 > 12%），实测把封面题注判成了表格：

        中华人民共和国个人所得税法 1980年9月10日 第五届全国人民代表大会…

    这一行有 3 组数字，密度 18%，但它是标题不是表格。区别在于**表格有行结构**：
    至少两行各含 ≥2 组数字（金额、税率、幅度）。题注只有一行带日期。
    """
    t = text or ""
    if t.count("|") >= 4:
        return True
    if len(t) <= 40:
        return False
    lines = [ln for ln in t.split("\n") if ln.strip()]
    rows = sum(1 for ln in lines if len(re.findall(r"[0-9０-９]+", ln)) >= 2)
    return rows >= 2


def supplement_stats(
    chunks: Sequence,
    prefer: Sequence[str] = DEFAULT_PREFER,
) -> Dict[str, int]:
    """法级让位到底丢了什么 —— 建库时打印的收益表。

    为什么要有这个函数（2026-11-13）
    ────────────────────────────────
    `drop_shadowed_laws` 按「法」整部让位，逻辑上必然有代价：md 漏收的条文会
    一起丢掉。但"有代价"不等于"这份代价是真的"——必须量出来，否则下次讨论
    「要不要改成按条补充」只能靠猜。

    实测（53 部 md + 41 部 pdf，9,760 块）：

        被让位丢掉的 pdf 块共 6,949 块
          · 6,854 块 → 条号在 md 里已有（真重复）
          ·    95 块 → 无条号的封面题注（标题 + 通过日期），零检索价值
          ·     0 块 → md 没有的独有条文

    也就是说：**按条补充的收益是 0**。所以保留法级让位，只把这个表打出来；
    以后语料变了（新增一部法、或 md 版本变旧）这张表会立刻显现，再决定是否改。

    分类口径：条号在 md 集合内 = `dup_article`；不在 = `extra_article`（真补充）；
    无条号再按 `_looks_like_table` 分 `table_like` / `prose`。
    """
    md_articles: Dict[str, set] = {}
    for c in chunks:
        meta = c.metadata or {}
        if _ext(meta) not in prefer:
            continue
        art = str(meta.get("article") or "").strip()
        if art:
            md_articles.setdefault(
                law_name(str(meta.get("source_file") or "")), set()).add(art)

    out = {"laws_covered": len(md_articles), "dup_article": 0,
           "extra_article": 0, "table_like": 0, "prose": 0}
    for c in chunks:
        meta = c.metadata or {}
        # 只统计 pdf：让位只作用在 pdf 上，图片另走按条去重（见 drop_shadowed_laws）
        if _ext(meta) != ".pdf":
            continue
        law = law_name(str(meta.get("source_file") or ""))
        if law not in md_articles:
            continue
        art = str(meta.get("article") or "").strip()
        if art:
            out["extra_article" if art not in md_articles[law] else "dup_article"] += 1
        elif _looks_like_table(c.text):
            out["table_like"] += 1
        else:
            out["prose"] += 1
    return out


def drop_toc_blocks(chunks: Iterable, max_chars: int = 400) -> Tuple[List, int]:
    """丢掉"目录块"。

    PDF 的目录页抽取出来长这样：
        `目录第一章 总 则第二章 授予专利权的条件第三章 专利的申请…`
    它没有条号、没有语义、但塞满了「第X章」字样 —— 问"第几章讲什么"时会被
    当成证据召回，答案就开始胡编章节内容。判据三条同时成立才丢：
    含「目录」、长度短（目录块通常不长）、「章」出现多次。
    """
    kept, dropped = [], 0
    for c in chunks:
        text = c.text or ""
        hit = (
            "目录" in text
            and len(text) <= max_chars
            and text.count("章") >= 3
        )
        if hit:
            dropped += 1
            continue
        kept.append(c)
    return kept, dropped
