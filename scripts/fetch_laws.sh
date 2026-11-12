#!/bin/bash
# 拉取中文法律语料（真实公开数据，不是合成）
#
# 通道选择说明：这台机器上
#   github.com / api.github.com / codeload.github.com  → 通（200）
#   raw.githubusercontent.com                          → 12 秒超时（只有 IPv6 记录，被封）
# 所以走 codeload 的 tarball，而不是逐文件 raw 下载。
# tarball 43 MB（压缩后），3 MB/s 左右，约 20 秒。
#
# 用法：bash scripts/fetch_laws.sh
set -euo pipefail

DEST="${1:-/home/zh/data/laws_raw}"
URL="https://codeload.github.com/LawRefBook/Laws/tar.gz/refs/heads/master"

mkdir -p "$DEST"
cd "$DEST"

if [ -d Laws-master ] && [ "$(find Laws-master -name '*.md' | wc -l)" -gt 10000 ]; then
  echo "✅ 已存在：$DEST/Laws-master（$(find Laws-master -name '*.md' | wc -l) 篇，跳过下载）"
  exit 0
fi

echo "⬇️  下载 $URL"
curl -sL --retry 3 --retry-delay 3 --max-time 900 -o laws.tar.gz "$URL"
ls -lh laws.tar.gz

echo "📦 解压"
rm -rf Laws-master
tar xzf laws.tar.gz
echo "✅ 完成：$(find Laws-master -name '*.md' | wc -l) 篇 md"
echo "   下一步：python scripts/prepare_law_corpus.py"
