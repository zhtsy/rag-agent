#!/usr/bin/env bash
# regress_law.sh —— 法律问答回归的唯一入口
#
# 三套用例一起跑，报告里按测试集分开统计：
#   test_cases_law.json        47 条  md 主链路（定义式 / 条号 / 数值 / 拒答）
#   test_cases_law_mixed.json  10 条  表格 / 扫描件 OCR / PDF 独有条文
#   test_cases_table.json      10 条  税额表定向用例
#
# 为什么要三套一起：47 条只证明 md 链路，非 txt 链路（PDF、图片、表格）
# 得靠后两套兜底。合成一个命令，是为了改完代码没有"忘记跑哪套"的余地。
#
# 用法：
#   bash scripts/regress_law.sh                     # 默认打 8020
#   bash scripts/regress_law.sh http://127.0.0.1:8010
#   bash scripts/regress_law.sh -- --limit 5        # 后面参数透传给 eval_system.py
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE="http://127.0.0.1:8020"
if [ "${1:-}" ] && [ "${1#-}" = "$1" ]; then
  BASE="$1"
  shift
fi
[ "${1:-}" = "--" ] && shift

PY="${PY:-/home/zh/myenv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M)"
OUT_DIR="$ROOT/reports"
mkdir -p "$OUT_DIR"

echo "▶ 回归：law 47 + mixed 10 + table 10 ｜ 后端 $BASE"
# -u：不缓冲。否则 stdout 接管道/tee 时是块缓冲，跑十几分钟都看不到进度。
"$PY" -u eval_system.py \
  --cases test_cases_law.json test_cases_law_mixed.json test_cases_table.json \
  --base "$BASE" \
  --sleep 0.3 \
  --out "$OUT_DIR/law_regress_${STAMP}.json" \
  --layer-report "$OUT_DIR/law_regress_${STAMP}.layer.md" \
  "$@"
