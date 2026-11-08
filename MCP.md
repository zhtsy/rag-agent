# MCP 接入

把本地知识库接进任何支持 MCP 的客户端（Claude Desktop、Cursor、自研 agent），
外部 agent 可以直接检索和问答，不用自己实现一套 RAG。

---

## 0. 它是怎么工作的

```
MCP 客户端 ──stdio JSON-RPC──> mcp_server.py ──HTTP + X-API-Key──> backend :8000
                                                                 ├── POST /search
                                                                 └── POST /chat/stream
```

`mcp_server.py` **不加载任何模型**，它只做三件事：翻译协议、翻译格式、翻译错误。
检索/排序/缓存/鉴权全在 backend 里，一行都没重写。

为什么必须这样：backend 已经常驻向量模型 + reranker 约 1.2G，这台机器只有 3.6G 内存。
MCP 进程若自己 import `rag_agent`，会再加载一份模型，直接 OOM；而且两份模型的
检索结果还可能因为缓存不同而不一致。

---

## 1. 前置条件

后端在跑（`docker compose ps` 里 backend 是 healthy），且 MCP 依赖已装：

```bash
cd /root/rag_deploy
pip3 install --target .mcp_libs mcp httpx      # 已装可跳过
```

> 为什么不用 venv：这台机器的 `python3-venv` 缺失（`ensurepip` 不可用），
> `python3 -m venv` 直接失败。`--target` + `PYTHONPATH` 是等效且更轻的做法。

依赖目录已在 `.gitignore` 里，不会入库。

---

## 2. 三种接法，按场景选

### A. stdio + SSH —— 最省事，推荐

客户端在本机，让它通过 SSH 拉起服务器上的进程。不需要开端口、不需要隧道。

```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "ssh",
      "args": [
        "-T", "root@47.239.14.225",
        "cd /root/rag_deploy && PYTHONPATH=.mcp_libs python3 mcp_server.py"
      ]
    }
  }
}
```

要求：本机到服务器已配好免密登录（SSH key）。

### B. SSE + 隧道 —— 多个客户端共用，或客户端只支持 URL

服务器上先起常驻服务：

```bash
cd /root/rag_deploy
PYTHONPATH=.mcp_libs python3 mcp_server.py --transport sse --port 8765
```

本机建隧道（**不要把 8765 直接开到公网**，见第 6 节）：

```bash
ssh -N -L 8765:127.0.0.1:8765 root@47.239.14.225
```

客户端配置：

```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "url": "http://127.0.0.1:8765/sse"
    }
  }
}
```

### C. 直接在服务器上跑（调试用）

```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "python3",
      "args": ["/root/rag_deploy/mcp_server.py"],
      "env": { "PYTHONPATH": "/root/rag_deploy/.mcp_libs" }
    }
  }
}
```

### 配置项（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAG_BASE_URL` | `http://127.0.0.1:8000` | 后端地址 |
| `RAG_API_KEY` | 空 | 优先用它；不设则读 `RAG_API_KEY_FILE` 或同目录 `.rag_api_key` |
| `RAG_TIMEOUT` | `180` | 单次请求超时（秒）。无实体过滤的检索要 10s+，别设太小 |
| `MCP_TRANSPORT` | `stdio` | `stdio` / `sse` / `streamable-http` |
| `MCP_PORT` | `8765` | SSE 端口 |

> 密钥默认复用后端那个 `.rag_api_key` 文件——**不新建第二份配置**。
> 轮换密钥时只改一处，不会出现"后端换了、MCP 还在用旧的"这类事故。

---

## 3. 两个工具

只暴露两个。工具越多模型越容易选错，越需要"循环探索"来兜底——
这两个动词的边界足够清楚，不需要更多。

### `search_knowledge_base(query, top_k=5)`

**只要证据，不要答案。** 适合：先确认知识库有没有这条资料、自己推理比对、
需要原始出处。

- `query` 必须自包含。带"他/她/它"时先换成具体实体名——检索层不做指代消解。
- `top_k` 只能往下截（1~20）。上游选块数量由检索策略决定，要 20 条也拿不到。

耗时：图谱路径约 2.5–3.5s；无实体过滤的全量检索 10s+（未命中尤其慢，要跑完多跳才判落空）。

### `ask_knowledge_base(question, session_id="mcp")`

**要一个组织好的答案。** 适合直接转述给用户。

- 同一 `session_id` 内多轮共享上下文，代词可指代前文实体。
- **换话题请换 session_id**，否则历史实体可能被带进新问题。

---

## 4. 返回长什么样

真命中（图谱路径）：

```
知识图谱回答（3322 ms，未命中缓存）：
来源：test.txt
小智住在南京。
（图谱事实可信度 0.9994）
```

向量命中：

```
共检索到 3 条证据（1180 ms，未命中缓存）：

[1] 来源：knowledge_base/人物志.md ｜ 相关度 0.9312
小智住在南京，喜欢猫。

[2] 来源：知识图谱属性 ｜ 确定性事实
加里奥由禁魔石制成
```

**未命中——注意这段是写给 agent 的指令，不只是数据：**

```
知识库中没有与这个问题相关的内容（10816 ms，未命中缓存）。

这不是服务故障，而是知识库本身没有覆盖这个话题。请如实告知用户
「知识库里没有这方面的资料」，不要换一种说法反复检索，也不要凭常识编造答案。
```

这句话是刻意设计的。不加它，上层 agent 会把一次"知识库没有"放大成五六次
无效检索，既费时间又让用户觉得系统答不出来还在硬撑。

**出错——指令和未命中相反：**

```
检索失败：无法连接知识库服务（http://127.0.0.1:8000）。请确认后端已启动。
这是服务端故障，不是知识库缺失，可以重试；若连续失败，请检查后端服务状态（/health）。
```

`no_match`（正常，别重试）和 `error`（故障，可以重试）**必须给出相反的指令**，
所以 `evidence.py` 把它俩拆成两个 kind，不允许含糊。

### 关于 `相关度`

它是最相关那条证据的 **rerank 分**，不是"答对的概率"。别拿它做业务阈值。
质量监控请看 `/feedback` 的 bad 率。

---

## 5. 怎么验证它真的能用

```bash
cd /root/rag_deploy
PYTHONPATH=.mcp_libs python3 mcp_smoke.py
```

它会像真正的客户端那样：起子进程、发 JSON-RPC 帧、按 id 收响应。覆盖 11 项：

- `initialize` 协商、capabilities 声明
- `tools/list`（工具名、参数 schema、描述非空）
- 两个工具的真实调用
- **stdout 未被污染**——往 stdout 里 print 一行就会毁掉 JSON-RPC 流

单测（`tests/test_mcp_server.py`，30 条）只覆盖逻辑；协议层靠这个脚本。
两件事都要做，它们抓的不是同一类问题。

---

## 6. 边界与已知问题

1. **SSE 模式没有鉴权。**
   MCP 客户端对自定义 header 的支持不统一，硬塞 token 会让很多客户端连不上。
   所以默认只监听 `127.0.0.1`，跨机使用请走 SSH 隧道。
   **不要把 8765 开到公网**——那等于一个免鉴权的知识库读取接口。

2. **无实体过滤的检索要 10s+。**
   这是既有检索策略的代价（要多跳跑完才判定落空），不是 MCP 层的问题。
   `RAG_TIMEOUT` 默认给 180s，留足余量。

3. **改完 Python 模块必须重启 backend。**
   热更新对已加载的模块无效。判据：看返回文本里有没有本该被剥掉的标签
   （比如 `【图谱检索结果】`）——有就是旧模块还在跑。
