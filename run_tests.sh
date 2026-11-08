#!/usr/bin/env bash
# 两层测试：一层在宿主机跑（纯逻辑、秒级），一层在容器里跑（需要后端依赖）。
#
# 为什么要分开：宿主机只装了 pytest，没有 langchain / neo4j driver / torch，
# 凡是 import 真实业务模块的测试都跑不起来。CI 同理（只装 pytest）。
# 所以分工是：
#   宿主机 / CI  → 纯函数与策略逻辑（confidence、retrieval_policy、缓存、归因…）
#   容器内       → 需要真实依赖的（图谱惰性连接、契约）
# 不把后者删掉，只是放在能跑它的地方跑。
set -uo pipefail
cd "$(dirname "$0")"

fail=0

echo "== 1/2 纯逻辑测试（宿主机）=="
if python3 -m pytest -q; then
    echo "   OK"
else
    echo "   FAIL"
    fail=1
fi

echo "== 2/2 需要后端依赖的测试（backend 容器内）=="
# 容器里没装 pytest（生产镜像不该带测试依赖），首次跑时补装；
# 容器 recreate 后会丢，脚本每次先探一下，避免"昨天还能跑今天跑不了"。
if ! docker compose exec -T backend python -c "import pytest" >/dev/null 2>&1; then
    echo "   容器内没有 pytest，临时安装（只影响当前容器，不进镜像）"
    docker compose exec -T backend pip install -q pytest \
        -i https://pypi.tuna.tsinghua.edu.cn/simple >/dev/null 2>&1
fi

if docker compose exec -T backend python -m pytest tests/test_lazy_graph.py -q; then
    echo "   OK"
else
    echo "   FAIL"
    fail=1
fi

exit "$fail"
