# -*- coding: utf-8 -*-
"""单问探针：问题以 base64 传入，答案以 base64 输出，避开中文转码。"""
import sys
import json
import time
import base64
import urllib.request

q = base64.b64decode(sys.argv[1]).decode("utf-8")
body = json.dumps({"message": q, "session_id": "probe-" + str(int(time.time()))}).encode("utf-8")
try:
    from api_auth import auth_headers
except Exception:  # noqa: BLE001
    def auth_headers(): return {}
req = urllib.request.Request(
    "http://127.0.0.1:8000/chat/stream",
    data=body,
    headers={"Content-Type": "application/json", **auth_headers()},
)

chunks, meta, t0 = [], {}, time.time()
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
            if "sources" in obj:
                meta["sources"] = obj["sources"]

ans = "".join(chunks)
src = meta.get("sources") or []
print("cost=%.1fs conf=%s len=%d src_n=%d" % (
    time.time() - t0, meta.get("confidence"), len(ans), len(src)))
print("B64:" + base64.b64encode(ans.encode("utf-8")).decode())
print("SRC:" + base64.b64encode(json.dumps(src, ensure_ascii=False).encode("utf-8")).decode())
