# -*- coding: utf-8 -*-
"""local_env.py —— 本地直跑时的运行环境预设（云端容器不生效）

为什么需要它
────────────
线上是容器部署：线程数、显存、联网策略由镜像和 compose 决定，代码里不需要关心。
本地是 WSL 直跑，下面这几项必须在 **import torch / transformers / FlagEmbedding 之前**
写进 os.environ，晚一步就没用了：

- `OMP_NUM_THREADS` / `MKL_NUM_THREADS`：不设的话 torch 会按物理核数开线程，
  2 vCPU 机器上是纯线程争抢，rerank 反而更慢。
- `TRANSFORMERS_OFFLINE` / `HF_HUB_OFFLINE`：离线机器上不设，sentence-transformers
  会卡在联网重试（表现是启动卡住几十秒，不报错，很难查）。
- `CUDA_VISIBLE_DEVICES`：本地旧版一直设 "0"，保持原状。

开关
────
只有 `.env` 里 `RAG_LOCAL_MODE=1` 时才生效，默认 0。
所以这份文件被带进云端也不会改变云端行为 —— 云端没有这个变量。

注意：本模块会在 import 期主动 load 一次 .env，比 rag_agent.py 里的 load_dotenv()
更早。重复加载是幂等的，无害。
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except Exception:  # noqa: BLE001  # 没有 dotenv 也不该让启动失败
    pass


def _enabled() -> bool:
    raw = (os.getenv("RAG_LOCAL_MODE") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _apply() -> None:
    if not _enabled():
        return

    # 线程数：只给 1，避免小机器上线程争抢
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # 离线模式：默认开，RAG_HF_OFFLINE=0 可以关掉（第一次下载模型时必须关）
    if (os.getenv("RAG_HF_OFFLINE") or "1").strip().lower() not in ("0", "false", "no", "off"):
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


_apply()
