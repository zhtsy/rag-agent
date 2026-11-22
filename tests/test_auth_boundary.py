# -*- coding: utf-8 -*-
"""鉴权边界回归：除了明确开放的端点，其余一律必须挂鉴权依赖。

为什么要单独一个文件
────────────────────
这个项目真实踩过的坑是"给主链路加了锁，旁边的门忘了关"：
`/chat/stream` 加了鉴权，但 `/feedback`（会把 question+answer+contexts 原样落盘）
和 `/feedback/stats` 一直是匿名可达；管理端 5001 也是全裸。
这类漏洞不会被功能测试发现——接口本身工作得很好，只是谁都能调。

所以这里不做行为测试，而是用 ast 读路由表：**新增端点默认必须鉴权**，
想开放就得显式写进 ALLOWED_OPEN / PUBLIC_BY_DESIGN，让"开放"成为一个需要解释的决定。

两种挂法都认
────────────
FastAPI 里鉴权可以写在装饰器上（`dependencies=[Depends(f)]`），
也可以写在函数签名上（`user: dict = Depends(f)`）。
早期只认第一种，结果新增的接口用第二种写法时这个测试会误报——
"测试只认自己熟悉的写法"会让它从护栏退化成噪音，所以两种都要解析。
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 三个受认可的鉴权依赖
GUARDS = {"require_api_key", "require_user", "require_chat_access"}

# 允许匿名的端点：根路径是存活展示，/health 给探针与负载均衡，/app 只是静态页跳转。
ALLOWED_OPEN = {"/", "/health", "/app"}

# 按定义必须匿名的端点——写在这里等于强制作者解释一次。
PUBLIC_BY_DESIGN = {
    "/auth/register": "注册接口按定义必须匿名（还没有账号）",
    "/auth/login": "登录接口按定义必须匿名",
    "/auth/logout": "登出只清自己的 Cookie，匿名调用无副作用",
    # 前端在登录页就要决定「显示不显示注册入口」，此刻还没有任何身份；
    # 它只回一个布尔值，不含任何账号信息，匿名不泄漏东西。
    "/auth/config": "登录前就要问的开关，只回 allow_register 布尔值",
}


def _routes(app_file: str = "main1.py"):
    """返回 [(函数名, 路径, 装饰器上的 dependencies 源文本, 签名里的 guard)]。"""
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
            out.append((
                node.name,
                route,
                ast.unparse(deps) if deps is not None else "",
                _param_guards(node),
            ))
    return out


def _param_guards(node) -> str:
    """扫函数签名的默认值，找出 `Depends(require_xxx)`。

    `kw_defaults` 里可能有 None（那是没有默认值的仅关键字参数），要跳过，
    否则 ast.unparse(None) 会返回字符串 "None"，把判断带偏。
    """
    defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
    found = []
    for d in defaults:
        src = ast.unparse(d)
        if "Depends" not in src:
            continue
        found += [g for g in GUARDS if g in src]
    return ",".join(found)


def _guard_text(name, route, deps, pguard) -> str:
    return f"{deps}|{pguard}"


def test_every_endpoint_is_guarded_unless_explicitly_open():
    routes = _routes()
    assert routes, "没解析到任何路由，说明装饰器写法变了，这个测试已经失效"

    unguarded = [
        route for (name, route, deps, pguard) in routes
        if route not in ALLOWED_OPEN
        and route not in PUBLIC_BY_DESIGN
        and not any(g in f"{deps}|{pguard}" for g in GUARDS)
    ]
    assert not unguarded, f"以下端点没有鉴权，也不是显式开放：{unguarded}"


def test_feedback_endpoints_are_guarded():
    """单独点名：这两个端点能读/写用户的问答原文，绝不能匿名。"""
    guarded = {
        r for (_, r, deps, pguard) in _routes()
        if any(g in f"{deps}|{pguard}" for g in GUARDS)
    }
    for path in ("/feedback", "/feedback/stats"):
        assert path in guarded, f"{path} 丢了鉴权依赖"


def test_conversation_endpoints_require_login_not_just_api_key():
    """会话归档是**用户私有数据**，只挂 require_api_key 是不够的：
    API-Key 是"这台机器可以调"，不是"这个人是这个账号"。
    """
    per_route = {}
    for (_, r, deps, pguard) in _routes():
        per_route.setdefault(r, "")
        per_route[r] += f"{deps}|{pguard}"
    for path in ("/conversations", "/conversations/{cid}", "/auth/me"):
        assert "require_user" in per_route.get(path, ""), f"{path} 必须要求登录"


def test_chat_accepts_either_login_or_api_key():
    """评测脚本只有 API-Key，浏览器只有 Cookie，两条通道都得留。"""
    deps_text = next((f"{deps}|{pg}" for (_, r, deps, pg) in _routes() if r == "/chat/stream"), "")
    assert "require_chat_access" in deps_text


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
