#!/usr/bin/env bash
# ablation_law.sh —— 存储/召回解耦 + 表格抽取 的消融实验
#
# 目的：把"我做了 X"变成"X 值多少分"。只有关掉某个机制再跑一遍，
# 才能回答面试里那句最要命的追问——"这个优化是你做的，还是碰巧？"
#
# 四组（唯一变量是检索侧的环境变量，索引库同一份，不重建）：
#   A base       默认            shadow 降权 0.05 + 表格块保留
#   B no-shadow  RAG_SHADOW_PENALTY=0   关掉"md 优先"        → 跑 47 条主集
#   C no-table   RAG_NO_TABLE=1          候选里剔除表格块     → 跑非 txt 集
#   D no-both    两个都关                                  → 跑非 txt 集
#
# 为什么 B 跑主集、C/D 跑非 txt 集：作用域本就不同。shadow 降权只在
# "同一部法 md 与 pdf 并存"时生效（29 部法），主集里这类题最多；表格块
# 只出现在 PDF/图片链路，非 txt 集才对它敏感。错配作用域会得出"没影响"
# 的假结论。
#
# 用法：bash scripts/ablation_law.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/zh/myenv/bin/python}"
DB="${RAG_CHROMA_DIR:-chroma_db_law_pdf}"
PORT=8020
OUT="$ROOT/reports/ablation"
mkdir -p "$OUT"

health() { curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000; }

start_backend() {
  local tag="$1"; shift
  fuser -k "$PORT/tcp" >/dev/null 2>&1 || true
  sleep 2
  (
    cd "$ROOT"
    # 关 tracing：LangSmith 限流时每次上传会阻塞重试，实测把 12s/题拖到 50s/题，
    # 一轮 67 条能从 13 分钟变成一小时。评测只关心答案与证据，不需要 trace。
    env RAG_CHROMA_DIR="$DB" RAG_EMBED_PROVIDER=ollama RAG_EMBED_MODEL=bge-m3 \
      LANGCHAIN_TRACING_V2=false LANGCHAIN_TRACING=false LANGSMITH_TRACING=false \
      "$@" \
      setsid "$PY" -m uvicorn main1:app --host 127.0.0.1 --port "$PORT" \
      > "/tmp/backend_ablation_$tag.log" 2>&1 < /dev/null &
  )
  for _ in $(seq 1 90); do
    if [ "$(health)" = "200" ]; then
      sleep 5
      echo "  ✓ 后端就绪 [$tag]"
      return 0
    fi
    sleep 2
  done
  echo "  ✗ 后端启动超时 [$tag]，见 /tmp/backend_ablation_$tag.log" >&2
  return 1
}

run_group() {
  local tag="$1" cases="$2"; shift 2
  if [ -f "$OUT/$tag.json" ]; then
    echo "→ 跳过 $tag（报告已存在：reports/ablation/$tag.json）"
    return 0
  fi
  start_backend "$tag" "$@" || return 1
  echo "→ 跑 $tag：$cases"
  # -u：不缓冲。否则 stdout 接管道时是块缓冲，跑一小时也看不到进度。
  # 各组另存一份 .log，主日志里只留当前组的尾部，避免四组输出混在一起。
  # shellcheck disable=SC2086
  "$PY" -u eval_system.py --cases $cases --base "http://127.0.0.1:$PORT" --sleep 0.3 \
    --out "$OUT/$tag.json" 2>&1 | tee "$OUT/$tag.log"
  echo
}

MAIN="test_cases_law.json"
NONTXT="test_cases_law_mixed.json test_cases_table.json"

run_group A "$MAIN $NONTXT"
run_group B "$MAIN" RAG_SHADOW_PENALTY=0
run_group C "$NONTXT" RAG_NO_TABLE=1
run_group D "$NONTXT" RAG_NO_TABLE=1 RAG_SHADOW_PENALTY=0

# 跑完把线上口径恢复成默认（不带消融变量）
fuser -k "$PORT/tcp" >/dev/null 2>&1 || true
echo "✅ 消融完成，结果在 reports/ablation/ ；8020 已停，需要时重新起默认后端。"
"$PY" scripts/ablation_summary.py || true
