# -*- coding: utf-8 -*-
"""auth.py —— 账号密码与登录令牌（不依赖第三方库）。

为什么不用 PyJWT / passlib
──────────────────────────
这两个库都只是把标准库包了一层：签名用 `hmac`，哈希用 `hashlib`。
为一次面试项目引入两个依赖，换来的是 requirements 里多两行和一次版本冲突风险，
不划算。这里用 stdlib 自己写，**格式显式写进文档**，反而更好讲。

密码怎么存
──────────
`scrypt$n$r$p$salt_hex$hash_hex`，`hashlib.scrypt(n=2**14, r=8, p=1)`。
选 scrypt 而不是 pbkdf2：scrypt 是**内存硬**的，攻击者用 GPU/ASIC 批量猜的
成本高一个量级；pbkdf2 只要算力，显卡堆得起来。代价是每次验证约 50~100ms，
对本项目的登录频率完全够用（登录不是热路径，问答才是）。

不存明文、不存可逆加密、salt 每人一份——三条都是底线，没有取舍空间。

令牌怎么签
──────────
`v1.<user_id>.<过期时间戳>.<签名>`
签名 = HMAC-SHA256(secret, "v1.<uid>.<exp>") 的 base64url。
无状态：服务端不存 session 表，重启不掉线；代价是**没法单条吊销**，
只能靠改 secret 全量失效。对"单机 + 自用"的场景，这个取舍是划算的；
真要吊销就把 exp 缩短到小时级。

secret 从哪来
─────────────
环境变量 `RAG_AUTH_SECRET`；没有就落在项目目录 `.auth_secret`（首次自动生成，
权限 600）。**不要把它提交进 git**——.gitignore 里已排除。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
SECRET_FILE = BASE_DIR / ".auth_secret"

COOKIE_NAME = "rag_token"
TOKEN_TTL = int(os.getenv("RAG_TOKEN_TTL", str(7 * 24 * 3600)))   # 默认 7 天
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1

# 用户名：3~24 个字符，字母/数字/下划线/连字符/中文。不用 \w，它会把日文假名也放进来。
_USERNAME_RE = re.compile(r"^[\w\u4e00-\u9fa5-]{3,24}$")
MIN_PASSWORD = 8


# =====================================================================
# 密码
# =====================================================================
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                        n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """验证密码。任何格式异常都返回 False，不抛——坏数据不能让登录接口 500。"""
    try:
        algo, n, r, p, salt_hex, hash_hex = stored.split("$")
        if algo != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
                            n=int(n), r=int(r), p=int(p), dklen=len(hash_hex) // 2)
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(dk.hex(), hash_hex)


def validate_credentials(username: str, password: str) -> Optional[str]:
    """校验注册输入，返回错误文案；合法返回 None。

    抽成纯函数是为了能单测——接口里的校验逻辑最难覆盖，一旦写成内联
    `if ... raise`，测试就得把整个 FastAPI app 拖起来。
    """
    if not _USERNAME_RE.match(username or ""):
        return "用户名需 3~24 位，仅限字母、数字、下划线、连字符或中文"
    if len(password or "") < MIN_PASSWORD:
        return f"密码至少 {MIN_PASSWORD} 位"
    if len(password) > 128:
        return "密码过长（上限 128 位）"
    return None


# =====================================================================
# 令牌
# =====================================================================
def _secret() -> bytes:
    env = (os.getenv("RAG_AUTH_SECRET") or "").strip()
    if env:
        return env.encode("utf-8")
    try:
        return SECRET_FILE.read_text(encoding="utf-8").strip().encode("utf-8")
    except OSError:
        pass
    raw = secrets.token_hex(32)
    try:
        SECRET_FILE.write_text(raw, encoding="utf-8")
        os.chmod(SECRET_FILE, 0o600)
    except OSError:
        pass
    return raw.encode("utf-8")


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def _sign(uid: str, exp: int) -> str:
    msg = f"v1.{uid}.{exp}".encode("utf-8")
    return _b64e(hmac.new(_secret(), msg, hashlib.sha256).digest())


def issue_token(user_id: str, ttl: int = 0) -> Tuple[str, int]:
    """返回 (令牌, 过期时间戳)。"""
    exp = int(time.time()) + (ttl or TOKEN_TTL)
    return f"v1.{user_id}.{exp}.{_sign(user_id, exp)}", exp


def parse_token(token: str) -> Optional[str]:
    """校验签名与有效期，返回 user_id；不合法返回 None。

    注意顺序：**先验签再读 exp**。反过来的话，攻击者改 exp 就能延长令牌，
    而签名只在最后才比对——这是一类经典的逻辑顺序错误。
    """
    try:
        ver, uid, exp_s, sig = (token or "").split(".")
    except ValueError:
        return None
    if ver != "v1" or not uid:
        return None
    try:
        exp = int(exp_s)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, _sign(uid, exp)):
        return None
    if exp < int(time.time()):
        return None
    return uid


def token_from_request(request: Any) -> str:
    """从 Cookie 或 Authorization 头取令牌。

    两种都支持不是为了"灵活"，是因为**两个调用方不一样**：
    浏览器带 Cookie（HttpOnly，JS 拿不到，XSS 偷不走），
    命令行/eval 脚本带 Bearer。只留一种就得让其中一方难受。
    """
    tok = ""
    try:
        tok = request.cookies.get(COOKIE_NAME) or ""
    except Exception:  # noqa: BLE001
        tok = ""
    if tok:
        return tok.strip()
    auth = ""
    try:
        auth = (request.headers.get("authorization") or "").strip()
    except Exception:  # noqa: BLE001
        auth = ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def user_from_request(request: Any) -> Optional[Dict[str, Any]]:
    import app_store  # 延迟导入：auth 的单测不该顺带建库

    uid = parse_token(token_from_request(request))
    if not uid:
        return None
    return app_store.get_user(uid)


# =====================================================================
# 登录失败限流
# =====================================================================
# 进程内字典。多副本时不精确（各算各的），但挡住脚本爆破足够了；
# 真要做准得上 Redis，那就超出这个项目的范围了。
_FAILS: Dict[str, list] = {}
_MAX_FAILS, _FAIL_WINDOW, _LOCK_SECONDS = 5, 300, 60


def login_blocked(key: str) -> int:
    """返回还需等待的秒数；0 表示没被锁。"""
    rec = _FAILS.get(key)
    if not rec:
        return 0
    count, first_ts, locked_until = rec
    now = time.time()
    if locked_until and now < locked_until:
        return int(locked_until - now) + 1
    if now - first_ts > _FAIL_WINDOW:
        _FAILS.pop(key, None)
        return 0
    return 0


def note_failure(key: str) -> None:
    now = time.time()
    count, first_ts, _ = _FAILS.get(key, (0, now, 0))
    if now - first_ts > _FAIL_WINDOW:
        count, first_ts = 0, now
    count += 1
    locked = now + _LOCK_SECONDS if count >= _MAX_FAILS else 0
    _FAILS[key] = [count, first_ts, locked]


def clear_failures(key: str) -> None:
    _FAILS.pop(key, None)


def allow_public_register() -> bool:
    """公开注册开关，**默认关闭**。

    默认关闭是有意的：8000 端口对公网开放，忘记配置就等于把注册权交给陌生人，
    他们能消耗 LLM 额度、往库里写数据。本地开发要注册就显式设
    `RAG_ALLOW_REGISTER=1`。

    放在这个模块而不是 main1：这里是纯逻辑，测试不必加载嵌入模型。
    """
    raw = (os.getenv("RAG_ALLOW_REGISTER") or "0").strip().lower()
    return raw not in ("0", "false", "no", "off")
