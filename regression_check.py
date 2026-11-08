# -*- coding: utf-8 -*-
"""回归检查：打 /chat/stream，远端判定关键词，答案 base64 回传。
避免中文经 workbench 回显时被转码。
"""
import json
import base64
import urllib.request

try:
    from api_auth import auth_headers
except Exception:  # noqa: BLE001
    def auth_headers(): return {}

BASE = "http://127.0.0.1:8000/chat/stream"

CASES = [
    ("Q1", "小陈养了什么宠物？", ["豆豆"]),
    ("Q2", "小张养的宠物叫什么名字？", ["布丁"]),
    ("Q3", "小李养了什么？", ["代码"]),
    ("Q4", "小明养了什么宠物？", ["没有", "未提", "没提", "不涉及", "没养", "没有提及"]),
    ("Q5", "今天天气怎么样？", None),
]


def ask(message, sid):
    body = json.dumps({"message": message, "session_id": sid}).encode("utf-8")
    req = urllib.request.Request(
        BASE, data=body, headers={"Content-Type": "application/json", **auth_headers()}
    )
    chunks, meta = [], {}
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if not p or p == "[DONE]":
                continue
            try:
                obj = json.loads(p)
            except Exception:
                continue
            if isinstance(obj, dict):
                if "text" in obj:
                    chunks.append(obj["text"])
                if "confidence" in obj:
                    meta["confidence"] = obj["confidence"]
    return "".join(chunks), meta


for qid, q, kws in CASES:
    try:
        ans, meta = ask(q, "regr-" + qid)
        conf = meta.get("confidence")
        hit = "" if kws is None else str(any(k in ans for k in kws))
        b = base64.b64encode(ans.encode("utf-8")).decode()
        print("%s|conf=%s|len=%d|kw=%s|b64=%s" % (qid, conf, len(ans), hit, b), flush=True)
    except Exception as e:
        print("%s|ERROR|%s: %s" % (qid, type(e).__name__, e), flush=True)
