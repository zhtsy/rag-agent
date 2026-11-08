# -*- coding: utf-8 -*-
"""鉴权边界回归：除了明确开放的端点，其余一律必须挂 require_api_key。

为什么要单独一个文件
────────────────────
这个项目真实踩过的坑是"给主链路加了锁，旁边的门忘了关"：
`/chat/stream` 加了鉴权，但 `/feedback`（会把 question+answer+contexts 原样落盘）
和 `/feedback/stats` 一直是匿名可达；管理端 5001 也是全裸。
这类漏洞不会被功能测试发现——接口本身工作得很好，只是谁都能调。

所以这里不做行为测试，而是用 ast 读路由表：**新增端点默认必须鉴权**，
想开放就得显式写进 ALLOWED_OPEN，让"开放"成为一个需要解释的决定。
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 只有这两个端点允许匿名：根路径是存活展示，/health 要给探针和负载均衡用。
ALLOWED_OPEN = {"/", "/health"}


def _routes(app_file: str = "main1.py"):
    """返回 [(函数名, 路径, dependencies 源文本)]。"""
    tree = ast.parse((ROOT / app_file).read_text(encoding="utf-8"))
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in node.decorator_list:
            if not (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)):
                continue
            if not (isinstance(d.func.value, ast.Name) and d.func.value.id == "app"):
                continue
            route = d.args[0].value if d.args and isinstance(d.args[0], ast.Constant) else ""
            deps = next((k.value for k in d.keywords if k.arg == "dependencies"), None)
            out.append((node.name, route, ast.unparse(deps) if deps is not None else ""))
    return out


def test_every_endpoint_is_guarded_unless_explicitly_open():
    routes = _routes()
    assert routes, "没解析到任何路由，说明装饰器写法变了，这个测试已经失效"
    unguarded = [r for (_, r, deps) in routes
                 if r not in ALLOWED_OPEN and "require_api_key" not in deps]
    assert not unguarded, f"以下端点没有鉴权，也不是显式开放：{unguarded}"


def test_feedback_endpoints_are_guarded():
    """单独点名：这两个端点能读/写用户的问答原文，绝不能匿名。"""
    guarded = {r for (_, r, deps) in _routes() if "require_api_key" in deps}
    for path in ("/feedback", "/feedback/stats"):
        assert path in guarded, f"{path} 丢了鉴权依赖"


def test_streamlit_sends_token_to_feedback():
    """前端加了鉴权就必须跟着带钥匙，否则点👍👎全部 401（表现为"反馈提交失败"）。"""
    src = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert "def _auth_headers()" in src
    assert src.count("_auth_headers()") >= 3, "统计与反馈两处调用都要带鉴权头"
    assert "headers=_auth_headers()" in src


def test_admin_app_has_global_guard():
    """管理端能导出全量对话记录，必须是 fail-closed 的全局拦截。"""
    src = (ROOT / "admin_app.py").read_text(encoding="utf-8")
    assert "@app.before_request" in src
    assert "compare_digest" in src, "密钥比较要用常量时间比较，别用 ==（会泄露前缀匹配长度）"
