"""README.md 与 RESULTS.md 的数字一致性检查（纯标准库，不加载任何模型）。

## 为什么要这个脚本

README 是给人看的，RESULTS.md 是给证据看的，两份都会写分数。
历史上它们分叉过不只一次：

  - 38 题在 README 写着 0.9474，RESULTS 已经是 1.0000（补 18 条边之后没回头改 README）
  - pytest 数字同一份 README 里出现两个值（正文 122、代码块 118），RESULTS 是 332
  - 已知问题里"T031/T032 是设计取舍，暂不动"和 RESULTS 的 38/38 直接矛盾

人肉同步不可靠，所以做成可执行检查。

## 用法

    python3 check_docs.py        # 退出码 0 = 一致，1 = 有分叉
    python3 check_docs.py -v     # 逐项打印，包括通过的那些

## 检查项

1. 每个用例集的「得分 / 命中数 / 总题数」两份文档必须相同；
2. 每份文档内部必须自洽：命中数 / 总题数 == 得分，题数标注 == 总题数；
3. pytest 通过数必须相同；
4. 已经过期的表述（见 STALE）不许再出现在 README 里。

## 故意不做的事

不比对"历史值"（如 `补边前 0.9474`）。那些数字本来就该留在原文里解释演变，
把它们也纳入比对，只会逼着人删掉上下文。
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 两边表格列名不同，正则各写一条（README: "38 条"；RESULTS: "38 题"）。
CASE_RE_README = re.compile(
    r"\|\s*`(test_cases\d?)\.json`\s*（\s*(\d+)\s*条[^|]*\|"
    r"\s*\*\*([\d.]+)\*\*\s*（\s*(\d+)\s*/\s*(\d+)\s*）"
)
CASE_RE_RESULTS = re.compile(
    r"\|\s*`(test_cases\d?)`\s*（\s*(\d+)\s*题[^|]*\|"
    r"\s*\*\*([\d.]+)\*\*\s*（\s*(\d+)\s*/\s*(\d+)\s*）"
)

PYTEST_RE_README = re.compile(r"\|\s*`?pytest`?\s*\|[^|]*\*\*(\d+)\s*passed")
PYTEST_RE_RESULTS = re.compile(r"\|\s*宿主机\s*pytest\s*\|[^|]*\*\*(\d+)\s*passed")

# 这些表述已经被现实推翻，留着就是误导。
STALE = [
    "118 passed",
    "122 条通过",
    "0.3 秒跑完",
]

SCORE_TOL = 0.0005     # 0.9474 这种四舍五入到 4 位，容差要够
RATIO_TOL = 0.0005


def _read(name):
    path = ROOT / name
    if not path.is_file():
        raise SystemExit(f"找不到 {name}，请在项目根目录运行本脚本")
    return path.read_text(encoding="utf-8")


def parse_cases(text, pattern):
    """→ {test_cases2: {n, score, hit, total}}"""
    out = {}
    for m in pattern.finditer(text):
        key, n, score, hit, total = m.groups()
        out[key] = {
            "n": int(n),
            "score": float(score),
            "hit": int(hit),
            "total": int(total),
        }
    return out


def fmt(c):
    return f"{c['score']:.4f} ({c['hit']}/{c['total']})"


def main(argv):
    verbose = "-v" in argv or "--verbose" in argv
    readme = _read("README.md")
    results = _read("RESULTS.md")

    ok_all = True
    rows = []          # (ok, 标签, 说明)

    def check(ok, label, detail):
        nonlocal ok_all
        ok_all = ok_all and ok
        rows.append((ok, label, detail))

    a = parse_cases(readme, CASE_RE_README)
    b = parse_cases(results, CASE_RE_RESULTS)

    if not a or not b:
        check(False, "解析", f"README 抽到 {len(a)} 行，RESULTS 抽到 {len(b)} 行"
                             "（格式变了？正则要跟着改）")

    # 1) 集合：README 有的，RESULTS 也该有（反之说明 RESULTS 多写了没进 README 的集）
    for key in sorted(set(a) - set(b)):
        check(False, key, "只在 README 出现，RESULTS 里没有对应行")
    for key in sorted(set(b) - set(a)):
        check(False, key, "只在 RESULTS 出现，README 基线表漏了这个集")

    # 2) 逐集比对 + 各自内部自洽
    for key in sorted(set(a) & set(b)):
        ra, rb = a[key], b[key]
        same = (
            abs(ra["score"] - rb["score"]) <= SCORE_TOL
            and ra["hit"] == rb["hit"]
            and ra["total"] == rb["total"]
        )
        check(same, key,
              f"README {fmt(ra)} ｜ RESULTS {fmt(rb)}" if same
              else f"分叉：README {fmt(ra)} ≠ RESULTS {fmt(rb)}")

        for doc, c in (("README", ra), ("RESULTS", rb)):
            if c["hit"] == 0 and c["total"] == 0:
                continue
            # 命中/总题 必须等于得分
            if c["total"]:
                ratio = c["hit"] / c["total"]
                check(abs(ratio - c["score"]) <= RATIO_TOL, f"{key} @{doc}",
                      f"自洽性：{c['hit']}/{c['total']}={ratio:.4f} 与得分 "
                      f"{c['score']:.4f} 不符" if abs(ratio - c["score"]) > RATIO_TOL
                      else f"内部自洽 {c['hit']}/{c['total']} = {c['score']:.4f}")
            # 标注的题数必须等于分母
            check(c["n"] == c["total"], f"{key} @{doc}",
                  f"标注被扩成 {c['n']} 条，分母仍是 {c['total']}"
                  if c["n"] != c["total"] else f"题数标注 {c['n']} 与分母一致")

    # 3) pytest
    pa = PYTEST_RE_README.search(readme)
    pb = PYTEST_RE_RESULTS.search(results)
    if pa and pb:
        check(pa.group(1) == pb.group(1), "pytest",
              f"README {pa.group(1)} passed ｜ RESULTS {pb.group(1)} passed"
              if pa.group(1) == pb.group(1)
              else f"分叉：README {pa.group(1)} passed ≠ RESULTS {pb.group(1)} passed")
    else:
        check(False, "pytest", "至少一处没抽到（表格格式变了？）")

    # 4) 过期表述
    for s in STALE:
        check(s not in readme, "过期表述", f'README 里还有 "{s}"' if s in readme
              else f'无 "{s}"')

    for ok, label, detail in rows:
        if ok and not verbose:
            continue
        print(f"[{'OK' if ok else 'XX'}] {label:<16} {detail}")

    if ok_all:
        print(f"一致：{len(rows)} 项检查全部通过"
              + ("" if verbose else "（加 -v 看明细）"))
        return 0
    print("有不一致项，见上面 [XX] 行")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
