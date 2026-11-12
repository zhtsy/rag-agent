#!/bin/bash
# 诊断 WSL 出网能力：DNS / 直连 / 代理端口
set -u
GW=$(ip route | awk '/default/ {print $3; exit}')
echo "网关(Windows 宿主): ${GW:-未取到}"
echo "DNS 服务器:"; grep -v '^#' /etc/resolv.conf | tr -s ' '

echo
echo "=== DNS 解析 ==="
for h in github.com raw.githubusercontent.com api.github.com codeload.github.com gitee.com; do
  r=$(getent hosts "$h" 2>/dev/null | head -1 | awk '{print $1}')
  printf '  %-34s %s\n' "$h" "${r:-解析失败}"
done

echo
echo "=== Windows 宿主代理端口 ==="
found=0
for p in 7890 7891 7897 10808 10809 1080 2080 20171 8889 33210; do
  if timeout 2 bash -c "echo > /dev/tcp/$GW/$p" 2>/dev/null; then
    echo "  $GW:$p 开放"; found=1
  fi
done
[ "$found" = 0 ] && echo "  (宿主机上没有常见代理端口在监听)"

echo
echo "=== 直连 https ==="
for h in https://github.com https://raw.githubusercontent.com https://api.github.com; do
  printf '  %-38s ' "$h"
  if out=$(timeout 15 curl -s -o /dev/null -w 'code=%{http_code} t=%{time_total}s' --max-time 12 "$h" 2>&1); then
    echo "$out"
  else
    echo "失败: $out"
  fi
done

echo
echo "=== git remote 连得上吗（ls-remote 只看 head，不下载） ==="
cd /home/zh/py_code 2>/dev/null && timeout 20 git ls-remote --heads origin 2>&1 | head -3 || echo "  ls-remote 失败"
