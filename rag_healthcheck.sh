#!/usr/bin/env bash
# ============================================================
# RAG 系统健康检查脚本  v2
# 已在 i-j6c37h6yet1ospbdhzab / cn-hongkong 实测验证通过
#
# v2 修正（基于 2026-11-04 实测结果）：
#   1. 后端**没有** POST /chat 端点，只有 POST /chat/stream（SSE 流式）。
#      原 v1 直连 /chat 会得到 404，容易误判为"RAG 挂了"。
#   2. /chat/stream 返回 SSE，非普通 JSON，需逐个 data: 事件拼接。
#   3. 真实服务清单为 5 个：admin / backend / frontend / mysql / neo4j。
#
# 用法：bash rag_healthcheck.sh
# ============================================================
set -uo pipefail

APP_DIR="${APP_DIR:-/root/rag_deploy}"
API_BASE="${API_BASE:-http://127.0.0.1:8000}"
# 后端 /chat/stream 已加鉴权：密钥优先取环境变量 API_KEY，其次读项目目录下的 .rag_api_key
API_KEY="${API_KEY:-$(cat "$APP_DIR/.rag_api_key" 2>/dev/null | tr -d '\r\n')}"
SESSION_ID="${SESSION_ID:-test_cli}"
QUESTION="${QUESTION:-根据知识库，小陈养了什么宠物？}"
EXPECT_KEYWORD="${EXPECT_KEYWORD:-豆豆}"
CHECK_SERVICES="${CHECK_SERVICES:-frontend backend neo4j mysql admin}"

PASS=0; FAIL=0
ok()   { printf '\033[32m[PASS]\033[0m %s\n' "$*"; PASS=$((PASS+1)); }
ng()   { printf '\033[31m[FAIL]\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
info() { printf '\033[36m[INFO]\033[0m %s\n' "$*"; }

echo "============================================================"
echo " RAG 健康检查  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo " 目录: $APP_DIR    API: $API_BASE"
echo "============================================================"

[ -d "$APP_DIR" ] || { echo "致命: 目录不存在 $APP_DIR"; exit 1; }
cd "$APP_DIR" || exit 1

# ---------------- 测试 1：容器存活 ----------------
echo
echo "================ 测试 1：容器存活状态 ================"
PS_OUT=$(docker compose ps 2>&1)
echo "---- docker compose ps ----"
echo "$PS_OUT"
echo "---------------------------"

for svc in $CHECK_SERVICES; do
  line=$(printf '%s\n' "$PS_OUT" | grep -i -- "$svc" | head -1)
  if [ -z "$line" ]; then
    ng "$svc: 未找到容器"
  elif printf '%s' "$line" | grep -qi 'unhealthy'; then
    ng "$svc: unhealthy → $(printf '%s' "$line" | tr -s ' ' | cut -c1-100)"
  elif printf '%s' "$line" | grep -qiE '(^|[[:space:]])(Up|running)([[:space:]]|$)'; then
    ok "$svc: Up → $(printf '%s' "$line" | tr -s ' ' | cut -c1-100)"
  else
    ng "$svc: 异常 → $(printf '%s' "$line" | tr -s ' ' | cut -c1-100)"
  fi
done

# ---------------- 测试 2：API 功能 ----------------
echo
echo "================ 测试 2：API 功能（SSE 流式）================"

# 2.0 健康探针
HCODE=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' "$API_BASE/health" 2>/dev/null)
[ "$HCODE" = "200" ] && ok "GET /health → 200" || ng "GET /health → $HCODE"

# 2.1 路由自检：确认 /chat 是否存在
ROUTES=$(curl -sS -m 15 "$API_BASE/openapi.json" 2>/dev/null \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print(' '.join(sorted(d.get('paths',{}))))" 2>/dev/null)
info "可用路由: ${ROUTES:-<取不到 openapi>}"

ENDPOINT="$API_BASE/chat/stream"
if printf '%s' "$ROUTES" | grep -qw '/chat'; then
  ENDPOINT="$API_BASE/chat"
  info "检测到非流式 /chat，改用该端点"
elif printf '%s' "$ROUTES" | grep -qw '/chat/stream'; then
  info "未提供 /chat，使用 /chat/stream（本项目实际情况）"
else
  ng "既无 /chat 也无 /chat/stream，路由表异常"
fi

BODY_FILE=$(mktemp)
CODE=$(curl -sS -N -m 120 -o "$BODY_FILE" -w '%{http_code}' \
  -X POST "$ENDPOINT" -H 'Content-Type: application/json' -H "X-API-Key: $API_KEY" \
  -d "$(printf '{"message":"%s","session_id":"%s"}' "$QUESTION" "$SESSION_ID")" 2>/dev/null)
echo "HTTP 状态码: $CODE"

if [ "$CODE" != "200" ]; then
  ng "API 非 200（$CODE），响应体头部：$(head -c 300 "$BODY_FILE")"
else
  ok "API 返回 200"

  # 解析：既能处理 SSE，也能处理普通 JSON
  ANALYSIS=$(EXPECT_KEYWORD="$EXPECT_KEYWORD" python3 - "$BODY_FILE" <<'PY'
import json, os, re, sys
raw = open(sys.argv[1], encoding="utf-8", errors="replace").read()
kw = os.environ.get("EXPECT_KEYWORD", "")

if raw.lstrip().startswith("{"):
    try:
        d = json.loads(raw)
        print("MODE=json")
        ans = d.get("answer", "")
        conf, srcs = d.get("confidence"), d.get("sources") or d.get("references")
    except Exception as e:
        print("MODE=fail"); print("ERR=" + str(e)[:150]); raise SystemExit
else:
    print("MODE=sse")
    parts, conf, srcs, n = [], None, None, 0
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("", "[DONE]"):
            continue
        try:
            ev = json.loads(payload)
        except Exception:
            continue
        n += 1
        if isinstance(ev.get("text"), str):
            parts.append(ev["text"])
        if ev.get("confidence") is not None:
            conf = ev.get("confidence")
        if ev.get("sources") is not None:
            srcs = ev.get("sources")
        if isinstance(ev.get("contexts"), list):
            print("CONTEXTS=%d" % len(ev["contexts"]))
    ans = "".join(parts)
    print("EVENTS=%d" % n)

print("HIT=%s" % ("YES" if kw and kw in ans else "NO"))
print("ANSWER_LEN=%d" % len(ans))
print("CONFIDENCE=%r" % (conf,))
print("SOURCES=%s" % (len(srcs) if isinstance(srcs, list) else "none"))
print("ANSWER=" + ans.replace("\n", " ")[:500])
PY
)
  echo "---- 解析结果 ----"
  echo "$ANALYSIS"
  echo "------------------"

  if printf '%s' "$ANALYSIS" | grep -q '^MODE=fail'; then
    ng "响应无法解析：$(printf '%s' "$ANALYSIS" | sed -n 's/^ERR=//p')"
  else
    HIT=$(printf '%s\n' "$ANALYSIS" | sed -n 's/^HIT=//p')
    CONF=$(printf '%s\n' "$ANALYSIS" | sed -n 's/^CONFIDENCE=//p')
    ALEN=$(printf '%s\n' "$ANALYSIS" | sed -n 's/^ANSWER_LEN=//p')

    [ "$HIT" = "YES" ] \
      && ok "answer 含关键词「$EXPECT_KEYWORD」，检索→重排→生成链路通" \
      || ng "answer 不含「$EXPECT_KEYWORD」（answer 长度 $ALEN）"

    [ "${ALEN:-0}" -gt 0 ] || ng "answer 为空"

    CC=$(printf '%s' "$CONF" | tr -d "'\" ")
    if printf '%s' "$CC" | grep -qE '^[0-9]+(\.[0-9]+)?$'; then
      NV=$(python3 -c "c=float('$CC');print(round(c/100,4) if c>1 else round(c,4))")
      if python3 -c "import sys;sys.exit(0 if 0<=float('$NV')<=1 else 1)"; then
        if python3 -c "import sys;sys.exit(0 if float('$NV')>=0.6 else 1)"; then
          ok "confidence=$CONF（归一化 $NV）处于合理区间"
        else
          ng "confidence=$CONF（归一化 $NV）偏低 → 检索相关性不足"
        fi
      else
        ng "confidence=$CONF 超出合理量纲"
      fi
    elif [ "$CC" = "None" ] || [ -z "$CC" ]; then
      info "响应未携带 confidence 字段"
    else
      ng "confidence 非数值: $CONF"
    fi
  fi
fi
rm -f "$BODY_FILE"

# ---------------- 测试 3：日志 ----------------
echo
echo "================ 测试 3：日志与端口 ================"
for p in 8501:frontend 5001:admin 7474:neo4j 3306:mysql; do
  port=${p%%:*}; name=${p##*:}
  c=$(curl -sS -m 8 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/" 2>/dev/null)
  [ -n "$c" ] && [ "$c" != "000" ] && info "$name:$port 可达 (HTTP $c)" || info "$name:$port 未响应 HTTP（可能非 HTTP 协议）"
done

echo
echo "注意: docker compose logs 在非 TTY 下不输出 ANSI 颜色，"
echo "      「红色报错」需改用关键词匹配，不能靠颜色。"
echo
LOG_FILE=$(mktemp)
docker compose logs --tail=40 backend >"$LOG_FILE" 2>&1
HITS=$(grep -nEi 'traceback|error|exception|critical|fatal|refused|timeout|timed out|panic|oom|killed|unhealthy|connection reset|no such host| 5[0-9][0-9] ' "$LOG_FILE" || true)
WARNS=$(grep -nEi 'warning|warn' "$LOG_FILE" || true)

if [ -z "$HITS" ]; then
  ok "后端近 40 行日志无 error/exception/5xx"
else
  echo "---- 异常行 ----"; echo "$HITS"; echo "----------------"
  ng "发现 $(printf '%s\n' "$HITS" | wc -l | tr -d ' ') 行异常，需人工确认"
fi

if [ -n "$WARNS" ]; then
  echo "---- 警告行（通常非致命）----"
  printf '%s\n' "$WARNS" | cut -c1-200
  echo "------------------------------"
fi
rm -f "$LOG_FILE"

echo
echo "============================================================"
printf ' 汇总: \033[32mPASS=%d\033[0m  \033[31mFAIL=%d\033[0m\n' "$PASS" "$FAIL"
echo "============================================================"
[ "$FAIL" -eq 0 ] || exit 2
