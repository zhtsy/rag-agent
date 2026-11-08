"""文档一致性闸门：README.md 与 RESULTS.md 的分数不能分叉。

这里测的不是代码，是**两份文档之间的事实一致性**。
理由和 `test_contract.py` 一样：这类问题不会报错，只会让读的人被误导，
而"人肉同步"在本项目已经失效过一次（README 38 题停在 0.9474，RESULTS 已是 1.0000）。

检查逻辑在 `check_docs.py`，本文件只负责把它挂进 pytest / CI。
要单独看明细：`python check_docs.py -v`
"""

import check_docs


def test_readme_matches_results():
    rc = check_docs.main([])
    assert rc == 0, (
        "README.md 与 RESULTS.md 的数字对不上。\n"
        "跑 `python check_docs.py -v` 看逐项明细。\n"
        "如果是有意只改了一边，请把另一边也改掉——这两份文档对外都是'事实来源'。"
    )
