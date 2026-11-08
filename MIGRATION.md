# 云端 → 本地迁移说明（2026-11-07）

来源：`i-j6c37h6yet1ospbdhzab:/root/rag_deploy`（11-07 14:02 的最终版）
目标：`/home/zh/py_code`（WSL Ubuntu）

> 本文分两部分：**§一~§五 是代码层迁移**，**§六~§八 是数据层重建与实测**。
> 代码搬完只是"能 import"，真正跑通还要把向量库、图谱、MySQL 三条数据链路对齐。

---

## 一、结论先行

代码已全量同步，**但直接照搬云端代码本地跑不起来**，有四处必须改的地方，都已改好：

| # | 问题 | 不加会怎样 | 已改成 |
|---|---|---|---|
| 1 | 云端 embedding 是 `BAAI/bge-base-zh-v1.5`，本地是 Ollama `nomic-embed-text` | **最危险**：本地根本没有 bge 模型文件，且 HF 缓存为空 + 离线模式 → 启动即失败。若强行下载成功，则会用 bge 去查 nomic 建的库（同为 768 维、不报错、检索静默崩坏） | `embeddings_shared.get_embeddings()` 工厂，按 `RAG_EMBED_PROVIDER` 选实现 |
| 2 | 云端 `local_llm` 也写成了 `ChatDeepSeek`（放弃双模型） | 本地 Qwen 完全用不上，每次都走 API；断网直接不可用 | `RAG_LLM_PROVIDER=hybrid` 恢复双路：决策/复杂题走 DeepSeek，生成走 Qwen |
| 3 | 云端 reranker 用 `BAAI/bge-reranker-base`（HF id） | 本地 `~/.cache/huggingface` 不存在，离线模式下加载失败 | `RAG_RERANKER_PATH`，本地指向 `/home/zh/models/bge-reranker-base` |
| 4 | 云端顶层没有 `TRANSFORMERS_OFFLINE` / 线程数设置 | 离线机器上 sentence-transformers 卡联网重试几十秒；小机器线程争抢反而更慢 | 新增 `local_env.py`，`RAG_LOCAL_MODE=1` 才生效 |

代码改完只是第一步。**实际在 WSL 里跑起来之后，又暴露出 7 处问题**（决策模型错配、图谱模板写死 1 跳、
`backfill` 不加载 `.env`、6 个文件写死云端路径、chroma 相对路径、embedding 工厂读不到 `.env`、
测试收集被单个模块拖垮），见 §二 第 5–11 条。当前实测状态：**341 passed / 1 skipped，自检 28/28**。

---

## 二、改了什么（逐条）

### 1. Embedding 两种 provider

`embeddings_shared.py` 新增工厂：

```python
def get_embeddings():
    provider = (os.getenv("RAG_EMBED_PROVIDER") or "bge").strip().lower()
    if provider == "ollama":
        return OllamaEmbeddings(model=nomic-embed-text, base_url=...)
    return LocalEmbeddings(BAAI/bge-base-zh-v1.5)
```

默认值仍是 `bge`，所以**云端不设这个变量时行为与迁移前完全一致**。
本机 `.env` 里设的是 `RAG_EMBED_PROVIDER=ollama`。

调用点统一改成 `get_embeddings()`：`rag_agent.py`（建 vectorstore）、`bm25_retriever.py`、
`build_knowledge_base.py`（重建索引）、`run_ragas_eval.py`（评测）。

> ⚠️ **换 provider = 换向量空间 = 必须重建 chroma_db。**
> 两个模型都输出 768 维，chroma 不会报任何错，只会悄悄把检索质量打崩。

### 2. LLM 双路恢复

`rag_agent.py` 的模型初始化段改成按 `RAG_LLM_PROVIDER` 分叉：

```python
LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "deepseek")   # 云端默认
# hybrid: local_llm = ChatOllama(qwen2.5:7b)，local_llm_stream = 另一个不绑工具的 ChatOllama
# deepseek: local_llm = cloud_llm = ChatDeepSeek（= 云端行为）
```

`local_llm_stream` 在 hybrid 下**必须是另一个不 bind_tools 的实例**——
Ollama 一旦绑了工具就会吞掉 token 流（参数走 `tool_call_chunks`，正文等整轮结束），
前端表现是"转圈很久然后整段吐出"。这是本地旧版踩过的坑。

### 3. Reranker 路径

```python
RERANKER_PATH = os.getenv("RAG_RERANKER_PATH") or "BAAI/bge-reranker-base"
reranker = FlagReranker(RERANKER_PATH, use_fp16=False)
```

### 4. local_env.py

只做一件事：在 `import torch / FlagEmbedding` **之前**写 `os.environ`。
由 `RAG_LOCAL_MODE=1` 开关控制，云端没有这个变量 → 空转，行为不变。

它也会提前 `load_dotenv()`，因为 `rag_agent.py` 里的 `load_dotenv()` 在第 600 行左右，
等到那时再设 `TRANSFORMERS_OFFLINE` 已经晚了（`FlagEmbedding` 在 import 期就初始化了 HF）。

### 5. 决策模型与生成模型分离（实测才发现，最严重的一处）

**现象**：`谁养了猫` 整题被拒答；`金毛是什么品种` 返回 `sources=0` 但模型照样编了答案。

**根因**：本地 Qwen2.5-7B 在阶段 1 明确输出 `{"need_tool": false}`。
`decide_guard.should_force_retrieval()` 只在模型**沉默**（没给出任何判断）时兜底，
而"明确但错误的判断"是要被尊重的——这是上一轮修 T038 时定下的原则，不能反向推翻。

**修法**：不推翻判断，而是换判断者。`rag_agent.py:673 _pick_decide_llm()`，
阶段 1 决策固定走 DeepSeek，生成仍按复杂度路由到 Qwen：

```python
# RAG_DECIDE_PROVIDER=cloud（默认）→ 决策恒走 DeepSeek
# RAG_DECIDE_PROVIDER=auto        → 回到"决策也按复杂度路由"的旧行为
```

这正是"一半 DeepSeek 一半 Qwen"的原意。**云端 `RAG_LLM_PROVIDER=deepseek` 时该函数直接返回
云端实例，行为一行未变。**

### 6. 图谱模板把 `品种是` 写死成 1 跳

聚合题规则原本是 `-[:养了]->(pet)-[:品种是]->(b)-[:属于类别*0..1]->(cat)`。
但**本地旧库没有 `品种是` 这个关系类型**（云端有），整条链匹配 0 行：
graph 阶段跑了 899ms 返回空 → 向量也落空 → 最终拒答。

改成两跳都允许 0 跳：`-[:品种是*0..1]->`，两种库都兼容。规则 8 与示例 5 同步更新。

### 7. `backfill_breed.py` 不加载 `.env`，报告路径写死

- 默认连 `bolt://neo4j:7687`（compose 服务名），宿主机 DNS 直接失败 → 补
  `load_dotenv(Path(__file__).with_name(".env"))`
- 报告路径 `/root/logs`，而 `Path(...).is_dir()` 在无权限时抛 `PermissionError` 而不是返回 False → 改成 `RAG_LOG_DIR` 兜底到项目内 `logs/`

### 8. 一批写死的云端路径（6 个文件）

| 文件 | 写死内容 | 后果 |
|---|---|---|
| `streamlit_app.py` | 后端默认 `http://47.239.14.225:8000` | 本地起界面一点就打到云端，白烧额度 |
| `eval_gate.py` | `APP = Path("/root/rag_deploy")` | 读不到密钥，闸门一跑就 WARN |
| `injection_probe.py` `smoke_cache.py` `drill_degrade.py` `repro_decide.py` | 密钥/用例路径 `/root/rag_deploy/...` | 直接 FileNotFoundError |
| `test_mcp_e2e.py` | cwd 与 `PYTHONPATH=/root/mcp_lib` | 同上 |

统一改成 `Path(__file__).parent` 派生、`api_auth.api_key()` 取密钥、`RAG_LOCAL_MODE` 分叉。

### 9. chroma 用相对路径

`rag_agent.py` / `bm25_retriever.py` 里是 `persist_directory="./chroma_db"`。
换个目录启动会**静默新建一个空库**，检索全空且不报错。改成基于 `Path(__file__).parent` 的绝对路径。

### 10. embedding 工厂读不到 `.env`（最隐蔽的一处）

`get_embeddings()` 靠 `RAG_EMBED_PROVIDER` 选模型，但**命令行脚本没有任何人加载 `.env`**——
只有 `settings.py` 加载，而 `build_knowledge_base.py` 这类脚本不经过它。

于是工厂读不到变量 → 退回 `bge` → 离线环境卡在 HuggingFace 重试。
就算模型装得上更糟：**用 bge 建库、用 nomic 查询，两者都是 768 维，不报任何错，检索质量静默崩掉。**

修法：工厂内部兜底 `load_dotenv(项目根/.env)`，**仅在变量未设置时生效**，云端行为不变。

### 11. `test_mcp_server.py` 的 importorskip 粒度太粗

只 skip 了顶层 `mcp`，但本地装的是 MCP SDK 1.x，没有 `mcp.server.mcpserver` 子模块。
结果不是跳过，而是**收集期 CollectionError，整个测试套件崩掉**（一个模块拖垮 340 个用例）。

修法：追加 `pytest.importorskip("mcp.server.mcpserver")`。

### 12. 新增 `selftest_local.py`

24 项分五组：环境 / 依赖服务 / 资产 / 模型与检索 / 端到端。
`--fast` 跳过重模型加载，`--e2e` 额外测 SSE 流式与 401 鉴权。逐条输出 PASS/FAIL/SKIP 并汇总待修项。

---

## 三、文件处理明细

### 覆盖（两边同功能，用云端最终版，23 个）

`rag_agent.py` `main1.py` `graph_retriever.py` `bm25_retriever.py` `build_graph.py`
`build_knowledge_base.py` `streamlit_app.py` `eval_system.py` `mcp_server.py` `fix_source.py`
`generate_ragas_data.py` `run_ragas_eval.py` `print_all_chunks.py` `requirements.txt`
`docker-compose.yml` `eval_report.json` `test_cases.json` `test_cases2.json`
`test_mcp_client.py` `README.md` `Dockerfile` 等

### 新增（云端有、本地没有，71 个顶层文件 + 21 个目录文件）

**核心模块**：`sanitizer.py` `evidence.py` `confidence.py` `answer_guard.py` `api_auth.py`
`settings.py` `embeddings_shared.py` `retrieval_router.py` `retrieval_policy.py`
`retrieval_cache.py` `decide_guard.py` `obs.py` `qa_logger.py` `trace_stats.py`
`geo_hierarchy.py` `breed_hierarchy.py` `backfill_geo.py` `backfill_breed.py`

**评测/运维**：`eval_gate.py` `eval_layers.py` `verify_router.py` `regression_check.py`
`bench_concurrency.py` `drill_degrade.py` `mcp_smoke.py` `merge_eval_reports.py`

**探针**：`probe_ask.py` `probe_graph.py` `probe_decide_query.py` `probe_scope.py`
`repro_decide.py` `diag_context.py` `diag.py` `diag_geo2~5.py` `noise_vec.py` `noise_rerank.py`

**测试**：`tests/` 18 个 + `failures/` 3 个

**脚本/文档**：`backup.sh` `rag_healthcheck.sh` `restore_verify.sh` `run_tests.sh`
`setup_qa_cleanup.sh` `docker-compose.override.yml` `.env.example` `requirements-dev.txt`
`CASE_STUDY.md` `OPS.md` `PERF.md` `RESULTS.md` `VARIANCE.md` `MCP.md` `LAYER_REPORT.md`

### 迁移后本地新增的文件

`selftest_local.py`（自检工具）、`check_docs.py` + `tests/test_doc_consistency.py`（文档数字闸门，
见 §八）、`logs/breed_backfill.json`（补全报告）、
`graph_extract_cache.jsonl`（抽取缓存，**不要删**，见 §七）、`.rag_api_key`（本地另生成）。

**补齐的迁移遗漏**：`.github/workflows/ci.yml`。迁移时按文件后缀复制，**隐藏目录被整目录跳过**，
所以本地一开始没有 CI 配置（云端存在）。11-07 已补，并把里面"整套 75 条 / 0.2 秒"这类
过期注释改成了不带易漂移数字的描述。云端另外还有 `.mcp_libs/`（vendored MCP SDK）和
`.env.bak.*`，属于机器本地状态，**不需要迁移**。

### 保留不动

- `.env`（本地密钥，只追加不覆盖）
- `knowledge_base/`（本地含 `food/` `games/` 子目录，比云端包全；22 篇与云端字节级一致）
- `dify/` `skills/` `.git/` `.idea/` `.cursor/`
- 本地练习脚本：`main.py` `1.py` `2.py` `Pydantic*.py` 结构化输出练习系列等
- `requirements1.txt`（完整 pip freeze，含 sentence-transformers）
- 旧测试：`test_core_flow.py` `test_stream_clean.py` `test_stream_queue.py` `test_eval_system.py`
  （`pytest.ini` 里 `testpaths = tests`，不会被收集，也不影响运行）

### 有意不覆盖的

- **`.gitignore`**：改为合并。云端版缺 `chroma_db/`、`models/`、`dify/` 几条，
  直接覆盖会让 1 GB 以上的模型目录和向量库变成未忽略状态。
- **`.rag_api_key`**：云端打包时带出来的密钥，**没有复制**。本地已另生成一份（`openssl rand -hex 24`）。
- **`chroma_db/`**：迁移时保留本地旧库，后来按 §六 重建（不是覆盖，是重新生成）。

---

## 四、本地启动

### 环境事实（踩过的坑）

| 项 | 值 |
|---|---|
| 主 Python | `/home/zh/myenv/bin/python`（3.12.3，依赖齐全） |
| 系统 `python3` | **一个依赖都没有**，`import chromadb` 直接失败 |
| 运行身份 | 必须 `su - zh -c`，root 下没有依赖 |
| Neo4j | Docker 容器 `neo4j-graphrag`，7474/7687 |
| MySQL | Docker 容器 `rag_mysql_local`，3306 绑 `127.0.0.1` |
| Ollama | systemd 服务，已 `systemctl enable --now`（默认不自启会踩空） |

### 一条命令起全栈

这台 WSL **空闲会自动重启**（`uptime` 多次实测 0–3 min），进程会死、`/tmp` 会清空。
所以启动、构建、校验**必须写在同一段脚本里**，否则分几次调用会得到前言不搭后语的结果。

```bash
systemctl start ollama
docker start neo4j-graphrag rag_mysql_local
cd /home/zh/py_code
su - zh -c "setsid /home/zh/myenv/bin/uvicorn main1:app --host 127.0.0.1 --port 8000 \
  > /tmp/rag_backend.log 2>&1 < /dev/null &"
curl -s --retry 30 --retry-delay 2 --retry-connrefused http://127.0.0.1:8000/health
```

前端与管理台：

```bash
su - zh -c 'cd ~/py_code && /home/zh/myenv/bin/streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1'
su - zh -c 'cd ~/py_code && /home/zh/myenv/bin/python admin_app.py'    # 5001，无密钥 401
```

调用后端要带密钥：

```bash
curl -H "X-API-Key: $(cat .rag_api_key)" ...
```

`.env` 里本地新增/改动的东西：

```
RAG_LOCAL_MODE=1
RAG_HF_OFFLINE=1
RAG_LLM_PROVIDER=hybrid
RAG_DECIDE_PROVIDER=cloud          # 决策固定走 DeepSeek，见 §二.5
RAG_EMBED_PROVIDER=ollama
RAG_EMBED_MODEL=nomic-embed-text
RAG_RERANKER_PATH=/home/zh/models/bge-reranker-base
NEO4J_URI=bolt://localhost:7687    # 云端是 bolt://neo4j:7687
DATABASE_URL=mysql+pymysql://rag_user:***@127.0.0.1:3306/rag_qa?charset=utf8mb4
```

---

## 五、代码层回退

```bash
cd /home/zh/py_code
bash _migrate_backup_20261107_1430/restore.sh
```

它会：① 用备份覆盖回 22 个原件；② 删掉本次新增的文件；③ 不动 chroma/knowledge_base/练习脚本。

备份目录内容：`files/`（迁移前 22 个原文件，含 `.env`）、`MANIFEST.txt`（清单 + 迁移前 `git status` + HEAD）、`restore.sh`。
`git` 本身也是一层回退网（HEAD = `803f7e8`）。

**注意：`restore.sh` 只回退代码，不碰数据层。** 数据层的回退见 §六、§七。

---

## 六、数据层（一）：向量库重建

### 现象

| | 本地（迁移时） | 云端 |
|---|---|---|
| 块数 | **92** | **145** |
| 元数据键 | `entity` `source` | `entity` `source` **`source_file`** |

语料 22 篇**字节级完全一致**，切分代码也是同一份（chunk 200 / overlap 40）。
所以差异纯属**历史构建遗留**：本地那份是旧版脚本、旧参数、旧元数据方案产出的。

### 为什么必须重建

1. 缺 `source_file` 元数据 → 引用溯源只能给到文件名级，拿不到原始路径。
2. 块数少 53 块意味着部分语料没被索引到 → 召回上限被压低。
3. 旧库是旧版脚本的产物，**无法确认里面存的是哪个模型的向量**。查库只能看到 768 维的
   float32 blob，看不出模型身份；而维数相同恰恰掩盖了不匹配（§二.10）。
   查询侧已经确定是 nomic，那唯一能保证"建库与查询同空间"的办法就是重建。

### 操作

```bash
# 0. 先停后端 —— Chroma 同时被两进程写会锁库
pkill -9 -u zh -f "uvicorn main1"
# 1. 备份
cd /home/zh/py_code && tar czf /home/zh/chroma_db_backup_20261107.tgz chroma_db
# 2. 重建（脚本幂等，会先清空 collection）
su - zh -c 'cd /home/zh/py_code && /home/zh/myenv/bin/python build_knowledge_base.py'
```

### 结果

```
块数        145          （= 云端）
元数据键    chroma:document 145 / entity 145 / source 145 / source_file 145
```

**回退**：`tar xzf /home/zh/chroma_db_backup_20261107.tgz -C /home/zh/py_code`（会回到 92 块的旧库）。

---

## 七、数据层（二）：图谱重建

### 先说结论：做不到"逐条复刻"

重建前先做了关系类型分布对比，因为盲跑不一定得到想要的：

```
云端更多：  拥有 243 / 制作 90 / 由...制成 118
本地更多：  喜欢 103 / 养了 7 / 材料是 1
```

**双方都不是对方的超集。** 这说明差异来自"两次 DeepSeek 抽取的随机性"，
不是"本地漏了某一类"。所以目标只能是"把覆盖面补到云端量级"，做不到逐边一致——
跑第三次又会是另一组数字。

### 操作

```bash
su - zh -c 'cd /home/zh/py_code && /home/zh/myenv/bin/python build_graph.py'          # 全量抽 145 块
su - zh -c '... build_graph.py --reuse'                                                # 复用缓存，只写库
```

成本：145 个分块，**114 秒**，约 1.2 秒/块。缓存落在 `graph_extract_cache.jsonl`
（145 行 / 97 KB），**保留它**，以后重建不再花 API 钱。

### 结果

| | 重建前 | 重建后 | 云端 |
|---|---|---|---|
| 实体 `__Entity__` | 683 | **741** | 742 |
| 关系 | 692 | **882** | 895 |
| 类别节点 | 2 | 2 | 2 |

当前关系分布（合计 882）：

```
拥有 243 / 属于 242 / 由...制成 121 / 喜欢 100 / 制作 76 / 擅长 39
住在 21 / 不喜欢 16 / 工作于 15 / 养了 5 / 属于类别 2 / 职位是 1 / 材料是 1
```

### 清掉一处脏数据

重跑后实体数正好 742，但没有立刻收工，抽查 `养了` 时发现：

```
我 → 豆豆 / 多肉 / 布丁 / 代码 / 拿铁      ← 66 条边的假节点
```

`我` 被当成了人物节点。根因在 `build_graph.normalize()`：它**只在分块带 `entity` 元数据时**
才把第一人称"我"映射成实际主体，而 `food/` `games/` 这些分块没有主体元数据 → 漏映射。
结果四个人的宠物全被算到一个"我"头上。

云端没有这个节点。已 `DETACH DELETE`，`养了` 回到 5 条真实关系，无残留代词节点。

### 残留问题（不影响答案，但要知道）

本地 `养了` **5 条，云端 4 条**。多出的一条是 `小李 → 代码`：

```
小李 → 橘猫      小李 → 代码      ← 其实是同一只猫
```

语料 `test5.txt` 写的是「我养了一只橘猫，叫"代码"」——**品种和名字被拆成了两个实体节点**。
它不会污染聚合答案（`代码` 没有 `品种是` 边，走 `*0..1` 匹配 0 行，不进结果），
但会让"小李养了什么"这类直查多列一个宠物。要彻底修得给 `代码` 补 `品种是 → 橘猫`，
代价是聚合查询会把小李算两次（规则按 pet 维度返回）。当前保留原样。

**回退**：全靠 `MERGE` 只增不删，原来那 692 条边一条没丢；唯一显式删除的是"我"节点。
真要回到 692 条，按缓存重放后删掉新增边。**没有一键回退**——
原计划导快照，但 `cypher-shell --format plain` 在这个版本不支持，导出的是 0 行空文件。

---

## 八、实测验证（2026-11-07 晚）

### 静态层

- 单元测试：**341 passed / 1 skipped**（`/home/zh/myenv`，冷启动 5.0s；增量 3.7s）。
  跳过的那 1 条是 `test_mcp_server`——它要求 MCP SDK 2.x，本机装的是 1.x。
  （云端镜像 332 passed，差 8 条是 `test_lazy_graph`：本地依赖齐备才真正执行。）
- 121 个 py 文件语法通过；云端快照 80 个 py 本地全在，ast 比对**顶层函数/类零丢失**。
- 接口清单：本地 7 条路由，路径与方法与云端**完全一致**。

### 运行层

`selftest_local.py --e2e` → **28/28 PASS**：

| 项 | 实测值 |
|---|---|
| 401 鉴权 | 无 key 正确拒绝 |
| SSE 流式 | 7.6 s / 27 字 / `conf=0.9957` |
| embedding 维度 | 768（nomic） |
| reranker 判别力 | 正例 0.9987 vs 负例 0.0000 |
| 前端 8501 | 200 |
| 管理台 5001 | 401（鉴权生效） |
| MySQL | 写入 / 反馈关联 / 统计 全通 |

分路问答验证（六题）：

| 问法 | 路径 | 结果 |
|---|---|---|
| 小陈住在哪个省 | 图谱 | 0.9998 ✓ |
| 谁养了猫 / 养猫的是哪几个人 | 聚合→图谱 | 0.9819 ✓ 小李养了橘猫 |
| 金毛是什么品种 | 品种层级 | 0.5818，5 个来源 ✓ |
| 泰拉瑞亚是什么游戏 | 向量 | 0.831 ✓ |
| 你支持七天无理由退货吗 | 知识库外 | 正确拒答 ✓ |

### 文档层：README 与 RESULTS 不再靠人肉同步

迁移后核对 README 时发现它的评测基线**落后两代**：38 题还写着 0.9474（RESULTS 已是 1.0000）、
pytest 同一份文档里出现 118 和 122 两个值（实际 341）、已知问题第 11 条还写着
"T031/T032 是设计取舍，暂不动"（早已补边修复）。

这类问题不会让任何程序报错，只会让读的人被误导，而两份文档都对外声称是事实来源。
所以加了 `check_docs.py` + `tests/test_doc_consistency.py`：

```bash
python check_docs.py -v      # 逐项：每个用例集的得分/命中/总题、pytest 通过数、过期表述
pytest tests/test_doc_consistency.py -q
```

四条规则：① 两边同名用例集的得分/命中/总题必须相同；② 各自内部自洽（命中÷总题 == 得分、
标注题数 == 分母）；③ pytest 通过数相同；④ 过期表述（如 `118 passed`）不许再出现。

**故意不比对历史值**（如"补边前 0.9474"）——那些数字本该留在原文解释演变，
纳入比对只会逼人删上下文。

已验证能真的抓到问题，不是永远绿的摆设：改回 0.9474 → 报分叉；改 RESULTS 的 332 → 报 pytest 分叉；
改标注 38→39 → 报题数不一致。

### 备份位置

| 内容 | 路径 |
|---|---|
| 代码层原件（22 个） | `_migrate_backup_20261107_1430/` |
| 旧向量库（92 块） | `/home/zh/chroma_db_backup_20261107.tgz` |
| 图谱抽取缓存 | `graph_extract_cache.jsonl`（145 行） |

---

## 九、已知限制

1. **"实测通过"只覆盖本机 WSL**。341 passed、28/28 都是这台机器跑出来的；
   云端是 bge + DeepSeek，两边数字不可互相印证。

2. **`docker-compose.yml` 仍是云端形状**。frontend 指向不存在的 `./frontend_image`，
   直接 `docker compose up` 会失败。本地现在是"直跑进程 + 两个容器（neo4j / mysql）"。
   要做成本地全栈 compose，得先补这个构建目录。

3. **两份 `chroma_db` 不能互拷**。本地 nomic、云端 bge，同为 768 维，
   拷过去不报错但检索结果全乱。

4. **评测基线不可比**。云端 38 题 1.0000 是 bge + DeepSeek 的口径；
   本地换 nomic + Qwen 后实测 0.9737（37/38，见 §十）。差 1 题，且已定位到生成层。
   要严格对齐需切 bge 并重建索引 + 换 DeepSeek 生成。

5. **图谱不是云端副本**。741/882 vs 742/895，差异来自抽取随机性；另有 `小李→代码` 重复边（§七）。
   节点级一致做不到，只能说"同量级"。

6. **第一人称问题无法消歧**。`我养了什么宠物` 这类问法，语料是多人的第一人称文本，
   "我"指向不确定，云端同样不支持。这是数据限制，不是代码问题。

7. **`ragas` 两边都没装**，`run_ragas_eval.py` 走不了；评测实际用的是 `eval_gate.py` / `eval_layers.py`。

8. **`requirements.txt` 追加了一行 `sentence-transformers`**。云端镜像预装所以原本没列，本地跑 bge 路径必须有它。

9. **WSL 空闲自动重启**。`uptime` 反复出现 0–3 min，进程死、`/tmp` 清空、
   未 checkpoint 的 sqlite WAL 会丢（第一次重建后读到 0 块就是这个原因）。
   所有长任务必须"启动 + 构建 + 校验"在同一段脚本内完成。

## 十、本地 vs 云端：效果到底差多少（2026-11-07 晚实测）

问的是"本地换成本地模型后，效果会不会掉很多"。这个不能靠印象答，用同一套 38 题跑了一遍。

### 结论

| | 得分 | 说明 |
|---|---|---|
| 云端（bge + DeepSeek） | **1.0000**（38/38） | 2026-11-06 基线，见 `RESULTS.md` |
| 本地（nomic + Qwen2.5-7B，修前） | **0.9342**（35.5/38） | 2026-11-07 20:19 |
| 本地（修后） | **0.9737**（37/38） | 2026-11-07 20:25 |

**差 0.0263，也就是 1 道题。** 而且这 1 题不怪检索。

### 分主题对照

| 类别 | 题数 | 云端 | 本地（修前） | 本地（修后） |
|---|---:|---:|---:|---:|
| 直接检索 | 17 | 1.000 | 0.971 | **1.000** |
| 多跳推理 | 6 | 1.000 | 0.833 | **0.833** |
| 拒答 | 4 | 1.000 | 1.000 | **1.000** |
| 跨主题 | 3 | 1.000 | 0.833 | **1.000** |
| 图谱-关系 | 5 | 1.000 | 0.900 | **1.000** |
| 图谱-反向查询 | 1 | 1.000 | 1.000 | **1.000** |
| 图谱-多跳 | 2 | 1.000 | 1.000 | **1.000** |

拒答 4 题在本地也是满分——**换小模型没有让它变得爱瞎猜**，这条比总分更重要。

### 检索层不但没退步，还略好

| 指标 | 云端 | 本地 |
|---|---|---|
| 检索召回率（只看证据，不看生成） | 0.9412 | **1.0000** |

修前是 0.956，修后 1.000。原因见下：那 3 道题的证据其实一直都在，只是没被送去生成。

### 第一处丢分：3 道题同一个根因，且与模型无关

`T009` / `T029` / `T035` 是同一道题的不同问法（「小陈养了什么宠物」），判据要同时命中
「金毛」和「豆豆」。修前三题都答「小陈养了豆豆」，各得 0.5。

日志里看得很清楚：

```
⚡ 路由规则命中 rule:pet，跳过模型意图判断
MATCH (p {id: "小陈"})-[:养了]->(pet) RETURN properties(pet)
🧩 图谱命中 1 行，结构化事实: 小陈养了豆豆。
```

图谱命中了就直接返回，**向量检索根本没跑**（而向量检索的第一名恰好就是含「金毛」的那块，
距离 0.4506）。问题在 Cypher 只 `RETURN properties(pet)`——品种不在 pet 节点的属性里，
而在边上：`豆豆 -[:属于]-> 金毛`。

**修法（`graph_retriever.py`，提示词层，两端通用）**：

1. 示例 3 改成把品种带出来，并用 OPTIONAL MATCH 兜住没有品种边的宠物：
   ```cypher
   MATCH (p {id: "小陈"})-[:养了]->(pet)
   OPTIONAL MATCH (pet)-[:品种是|属于]->(kind)
   RETURN properties(pet) AS pet, properties(kind) AS kind
   ```
2. 新增规则 9，写明「品种挂在边上、不写会把金毛弄丢」，并说明两种关系名都要兼容
   （本地库是 `属于`，云端是 `品种是`，抽取阶段不统一）。

修完后三题全变 1.00，答案变成「小陈养了豆豆和金毛」。**9 题定向回归 + 38 题全量无一处回退。**

### 第二处丢分：唯一一处真的输给 DeepSeek

`T019`「加里奥为什么能吸收魔法」，判据要命中「禁魔石」。证据里明明有：

```
【来源：知识图谱属性】【确定性事实】
加里奥 由...制成 禁魔石
```

Qwen 的答案通篇讲"长年累月吸纳魔法让他活了起来""用庞大的石躯吸收魔法弹"——
**被 5 段叙述性语料带跑，完全没引用那条结构化事实**。

同一份证据直接喂给 DeepSeek 做对照（`temperature=0`，不经过任何检索）：

| 模型 | 答案 | 含禁魔石 |
|---|---|---|
| Qwen2.5-7B | 根据资料，加里奥能够吸收魔法的原因是长年累月地吸纳魔法让他活了起来…… | ✗ |
| DeepSeek | **加里奥由禁魔石制成**，长年累月吸纳的魔法让他活了起来，因此能用庞大的石躯吸收魔法攻击。 | ✓ |

所以这一题是**模型能力差距，不是代码或检索问题**。7B 在"结构化事实 vs 长叙述"并存时，
倾向于复述叙述、丢掉那个孤立的关键名词。这类错误换提示词能压一部分，但压不干净。

### 怎么读这三个数字

1. **0.9737 不能当成"比云端差 2.6%"**。38 题里 1 题就是 2.6%，样本太小，
   少答对一个专有名词和整体能力下降是两个量级的事。
2. **真正的差别在"多跳推理"这一类的鲁棒性**，不在覆盖面。事实型、图谱型、拒答题
   本地已经追平；一旦要"在长文本里挑出那个关键名词"，7B 会漏。
3. **要完全对齐只需换生成模型**：`RAG_LLM_PROVIDER=cloud`（全 DeepSeek）即可，
   代码一行不用改。反过来，如果坚持全本地，那 T019 这类就是可预期的上限。

### 复现方式

```bash
# 起服务后（见 §四）
su - zh -c 'cd ~/py_code && /home/zh/myenv/bin/python eval_system.py \
  --cases test_cases2.json --out logs/local_eval.json --sleep 1'
```

注意：**改完代码或提示词必须重启后端并清缓存**，否则会命中旧答案：
```bash
curl -X POST -H "X-API-Key: $(cat .rag_api_key)" http://127.0.0.1:8000/admin/cache/clear
```
实测就踩过——`T035` 因为缓存直接返回了上一轮的错误答案（耗时 0.5s，`cache: hit`）。

报告留档：`logs/local_eval_full.json`（修前 0.9342）、`logs/local_eval_fixed.json`（修后 0.9737）、
`logs/recheck_pet.json`（9 题定向）。`logs/` 已在 `.gitignore` 里，不进仓库。
