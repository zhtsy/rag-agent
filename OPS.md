# 运维手册

这份文档记录的是**配置为什么这么写**，不是操作步骤清单。
数字全部来自 2026-11-05 的实测，复现方式见每节末尾。

---

## 1. 资源上限与健康检查

配置在 `docker-compose.override.yml`（增量文件，不改原 compose，原文件含密钥）。

| 服务 | mem_limit | 实测稳态 | 健康检查 | start_period |
|---|---|---|---|---|
| backend | 2400m | 1.22–1.26 GiB | `GET /health` | 120s |
| neo4j | 800m | 410–420 MiB | `bash /dev/tcp/127.0.0.1/7687` | 40s |
| mysql | 700m | 371–381 MiB | 原 compose 自带 | — |
| frontend | 320m | 74 MiB（装依赖时 154） | `/_stcore/health` | 40s |
| admin | 160m | 47–48 MiB | `GET /` | 30s |

三个不是随便定的地方：

**为什么加 mem_limit。** 不是为了省内存，是为了**让 OOM 只杀一个容器**。没有 limit 时，内存打满由内核全局挑一个牺牲者，最坏情况会杀掉 mysql——一个只占 78MiB、和内存压力毫无关系的服务。

**为什么必须是 healthcheck，而不是只靠 `restart: unless-stopped`。** restart 策略只看"进程退没退出"。backend 线程池耗尽、driver 卡死时，进程还活着，docker 认为它健康，重启永远不会触发。加之前只有 mysql 有健康检查。

**为什么 neo4j 探端口而不是跑 Cypher。** 健康检查不该依赖口令（口令轮换就得改探针），也不该在库繁忙时再压一条查询上去。

**mysql 和 frontend 的初版给紧了。** 第一版 mysql 给 400m，实测稳态 378m，只剩 22m 余量；frontend 给 160m，pip 安装期间峰值 154m。两个都是"看起来够、其实随时会死"的配置，第二次调宽。**limit 要按峰值给，不是按稳态给。**

---

## 2. 备份与恢复

### 备份什么、不备份什么

| 数据 | 能否重建 | 策略 |
|---|---|---|
| MySQL（会话、反馈、评测记录） | **不能** | `mysqldump` 每日，重点 |
| Neo4j 图谱 | 能，`build_graph.py` 从源文件重建 | 备份源文件即可 |
| Chroma 向量库 | 能，源文件重新向量化 | 同上 |

所以 `backup.sh` 产出两个文件：`mysql_*.sql.gz`（不可重建的）和 `kb_*.tar.gz`（重建原料）。dump 不带 `--databases`，避免文件里出现 `USE \`rag_qa\``——恢复演练时一不留神就会写进生产库。

### 没验证过的备份等于没有备份

`restore_verify.sh` 把最新 dump 导入临时库 `rag_restore_test`，**逐表比对 `count(*)`**，跑完删除临时库。只比表数量是不够的：表建出来了但数据没进去，是最常见的"假成功"。

### crontab

```
0 3 * * *   backup.sh              # 每日备份，保留 7 天
0 4 * * 0   restore_verify.sh      # 每周日验证能恢复
0 5 * * *   eval_gate.py           # 每日评测门禁
```

---

## 3. 评测门禁

```bash
python3 eval_gate.py --min-score 0.90
```

三条判定，任一不满足 exit 1：服务健康 → 成功条数 == 总条数 → 均分 ≥ 门槛。

**为什么不进 GitHub Actions。** 端到端评测要起 uvicorn + Chroma + Neo4j + MySQL + 两个 bge 模型（约 1.2GB）。runner 上没有这些，硬塞只有两种结果：长期红色（假门禁），或者把评测改成不依赖模型（那就不是端到端了）。分工是：**CI 跑 118 条纯逻辑单测（0.2 秒），这道门放在有模型的地方每天跑。**

**缓存开着时，报告里的耗时不能信。** 门禁会打印 `NOTE 检索缓存已开启`。分数仍然可信（答案一样），但耗时不能和冷路径基线比。

---

## 4. 降级演练：停掉 Neo4j 之后会发生什么

### 结论先说

**是软降级，不是硬故障**：HTTP 200、答案正确，但耗时一度到 101 秒——比 500 更难排查，因为没有任何错误信号，只有"服务好像死了"。

### 数据

| 场景 | 状态码 | 总耗时 | graph 阶段 |
|---|---:|---:|---:|
| 正常（基线） | 200 | 3.6–4.6s | — |
| Neo4j 停 · **修复前** | 200 | **101.4s** | **97.9s** |
| Neo4j 停 · 收紧 driver 超时后 | 200 | 28.3s | 11.5s |
| Neo4j 停 · 再跳过无效重试 | 200 | 16.4s | **4.1s** |
| 恢复后 | 200 | 5.6s | — |

### 两次修复

**1. driver 超时不设，等于没有超时。** `Neo4jGraph` 默认按 driver 的宽松默认值走，Neo4j 停掉后要重试满 98 秒才把异常抛出来。改成：

```python
graph = Neo4jGraph(
    timeout=3,
    driver_config={
        "connection_timeout": 3,
        "connection_acquisition_timeout": 5,
        "max_transaction_retry_time": 3,
    },
    **neo4j_kwargs(),
)
```

正常查询是毫秒级，3 秒不会误伤；不可用时 3 秒内失败并降级。

**2. 连接层失败不该重试。** `_run_cypher` 原本对任何异常都"带着报错回去让模型改一版 Cypher"。Cypher 写错时这很值，但 Neo4j 连不上时，重试只是把同一个超时再等一遍，还多花一次 LLM 调用。现在区分开：

```python
if _FATAL_DB_ERRORS and isinstance(e, _FATAL_DB_ERRORS):
    logger.warning("🚫 图谱连接不可用，跳过重试，直接交给向量降级")
    return [], cypher, error
```

### 一个必须记住的坑

**第一次演练是无效的。** 停掉 Neo4j 后同一个问题 1.5 秒就答完了、答案一字不差——那是**缓存命中**，根本没走检索。重跑时换了没问过的问题才看到真实的 101 秒。

> 演练依赖的时候，一定要用没问过的问题，或者先清缓存。
> 缓存会同时掩盖"快"和"慢"，让故障看起来像正常。

### 剩余耗时的构成

16.4 秒里：graph 4.1s（已尽力）+ rerank 8.4s + retrieve 2.4s。rerank 那 8 秒是 CPU 限频问题（突发性能型实例），与 Neo4j 无关，见 `PERF.md`。

---

## 5. 安全边界与启动依赖（2026-11-05 晚：原 4 条全部处理）

这一节原来记着 4 条"没解决"。现在都处理了，并在下面写清**判据**——
判断修没修好，看命令输出，不看"我觉得应该好了"。

### 5.1 端口：对外只留 8000 / 8501

| 服务 | 端口 | 绑定点 | 为什么 |
|---|---|---|---|
| backend | 8000 | 0.0.0.0 | 用户入口，要对外 |
| frontend | 8501 | 0.0.0.0 | 用户入口，要对外 |
| admin | 5001 | **127.0.0.1** | 能导出全库、删记录、改标注 |
| mysql | 3306 | **127.0.0.1** | 数据面，只有本机需要 |
| neo4j | 7474 / 7687 | **127.0.0.1** | 数据面，只有本机需要 |

判据：

```bash
ss -lntp | grep -E ':(5001|3306|7474|7687|8000|8501)\b'
# 期望：8000/8501 是 0.0.0.0，其余四个是 127.0.0.1
```

要看管理后台（5001 已不对公网开放），走 SSH 隧道：

```bash
ssh -L 5001:127.0.0.1:5001 root@<服务器>
# 然后浏览器打开 http://127.0.0.1:5001/?key=<.rag_api_key 的内容>
```

### 5.2 管理端加了全站鉴权

`admin_app.py` 用 `before_request` 拦所有路由（默认拒绝，新增路由不会忘记加保护），
密钥来源与后端同一个 `api_auth.api_key()`，比对用 `hmac.compare_digest`。
密钥文件缺失时**也拒绝**（fail-closed）：这个台子能导出全库，宁可让运维去补密钥。

判据：

```bash
curl -o /dev/null -w '%{http_code}' http://127.0.0.1:5001/            # 401
curl -o /dev/null -w '%{http_code}' "http://127.0.0.1:5001/?key=错的"  # 401
curl -o /dev/null -w '%{http_code}' "http://127.0.0.1:5001/?key=$(cat .rag_api_key)"  # 200
```

顺带一个坑：加了鉴权后，原来的健康检查探 `/` 会拿到 401、被 `urlopen` 抛成
`HTTPError`，容器会被判 unhealthy。探针改成连 TCP 端口——健康检查判"进程活着"，
不判"能不能访问"。

### 5.3 frontend 依赖固化进镜像

原来它跑 `python:3.10-slim` + 启动命令 `pip install ... && streamlit run`，
每次 recreate 都要联网重装（约 2 分钟，期间 unhealthy），装的版本还随 pypi 漂移。
现在 `frontend_image/Dockerfile` 在构建期装好依赖，override 里给 frontend 加了
`build:` 并把 command 改成纯 `streamlit run`。

实测：`docker compose up -d` 之后 frontend 从启动到 healthy **不到 1 分钟**
（不再有 pip 安装段），mem_limit 也从 320m 降到 256m（不再需要覆盖安装期峰值）。

### 5.4 backend 启动不再依赖 Neo4j（本次最重要的一条）

原先 `graph_retriever.py` 在模块层就 `graph = Neo4jGraph(...)`，所以 **import 等于必须连上库**。
实测故障：

```
docker compose stop neo4j && docker compose restart backend
→ File "/app/graph_retriever.py", line 54, in <module>
  ValueError: Could not connect to Neo4j database
→ 容器 Up 2 seconds 后退出，进入重启循环，/health 无响应
```

这比"降级慢"严重得多：Neo4j 挂了，连不需要图谱的普通问答也一起没了，
而且**在 Neo4j 修好之前后端永远起不来**——没人能修，因为它自己就是被修的东西。

改法：惰性连接 + 专用异常 + 冷却。

- `get_graph()` 首次真正用到图谱时才建连接，失败抛 `GraphUnavailable`
- 失败后 15 秒内不再重连（否则每个请求都白等一次 3 秒连接超时）
- `GraphUnavailable` 并入 `_FATAL_DB_ERRORS`，`_run_cypher` 在生成阶段就短路：
  连不上时**不会**再让模型"改一版 Cypher"（那是白花一次 LLM 调用）

判据（两条都要过）：

```bash
# 1) Neo4j 停着也能起来，并且普通问答可用
docker compose stop neo4j && docker compose restart backend
curl -s http://127.0.0.1:8000/health          # 200
python3 drill_degrade.py down3 test_cases3.json C001
#    → status=200，答案正确；日志出现「🚫 图谱不可用，跳过生成与重试，直接降级」

# 2) Neo4j 恢复后不重启后端，自动接上
docker compose start neo4j
sleep 45 && python3 drill_degrade.py restored test_cases2.json T037
#    → 200，答案正确；日志出现「✅ Neo4j 连接已建立（惰性连接）」
```

回归测试：`tests/test_lazy_graph.py`（8 条），覆盖"拿代理不连库""失败抛专用异常"
"冷却期不重连""不白调 LLM"，以及一条**真实 import 子进程**用例
（NEO4J_URI 指向不存在的端口，import 必须成功）——模块顶层连接一旦回来，这条立刻红。

### 5.5 仍然存在的边界（这次没解决的，照旧列出来）

1. **Neo4j 停时首次提问会多等约 3 秒**（连接超时），之后 15 秒冷却期内不再等。
   这个代价是有意的：3 秒换"图谱挂了我还能答题"。
2. **管理端密钥走 URL 参数**（`?key=`），会进浏览器历史和 access log。
   纯内网/隧道场景可接受；要更严就改成 cookie 登录态。
3. **密钥轮换靠手动改文件**（改完立即生效，不用重启），没有自动轮换/过期。
4. **T031 / T032 那 2 分不追**：证据只写到"在广州/在南京"，是否推一步到省份，
   属于推理策略选择。为了 2 分让它学会外推，会把瞎猜风险一起放大。

### 5.6 凭据轮换与外置（本轮补的，P0 最后一项）

问题不是"口令弱"，是**口令写在会进 git 的文件里**：`docker-compose.yml` 明文写着
`NEO4J_AUTH=neo4j/<口令>`（教科书默认值；此处原文已脱敏）、MySQL root 与业务账号口令，还有 3 处内嵌口令的
`DATABASE_URL`；而仓库根目录**没有 `.gitignore`**。一旦提交，等于把生产凭据连同 DeepSeek / LangSmith
的 key 一起公开。

改动：
1. compose 8 处口令改为 `${VAR}`，由 `.env` 插值（`docker compose config` 通过，残留明文 0 处）。
2. Neo4j 与 MySQL 三处账号口令轮换为 32 位随机字母数字，库内同步执行
   `ALTER USER` / `ALTER CURRENT USER`。
3. 新增 `.gitignore`：`.env`、`.env.*`、`.rag_api_key`、`*.bak.*`、`__pycache__`。
4. `/feedback`、`/feedback/stats` 挂上 `require_api_key`，前端两处调用补 `X-API-Key`。

判据（2026-11-05 实测）：

| 检查 | 结果 |
|---|---|
| MySQL root 新口令 | OK |
| MySQL `rag_user` 新口令 | OK（容器内读 `qa_records` 5 行） |
| MySQL 旧 root 口令 | **rejected** |
| Neo4j 旧口令 | **rejected** |
| `POST /feedback` | 无 key **401** / 有 key 200 |
| `GET /feedback/stats` | 无 key **401** / 有 key 200 |
| `POST /admin/cache/clear` | 无 key **401** / 有 key 200（见 5.7） |
| `POST /search` | 无 key **401** / 有 key 200（MCP 工具的数据来源，见第 7 节） |
| `tests/test_auth_boundary.py` | 4 passed；把依赖去掉会立刻变红（反向验证过，不是空测试） |

轮换怎么做（下次照做，注意两个坑）：

1. 生成口令：只留字母数字，避开 URL 与 shell 转义，例如
   `python3 -c "import secrets,string;a=string.ascii_letters+string.digits;print(''.join(secrets.choice(a) for _ in range(32)))"`。
2. **改 `.env` 必须同时改库内口令**：MySQL `ALTER USER 'root'@'%' IDENTIFIED BY ...`；
   Neo4j `ALTER CURRENT USER SET PASSWORD FROM 'old' TO 'new'`。
   容器的 `MYSQL_ROOT_PASSWORD` / `NEO4J_AUTH` 只在**首次初始化空数据目录**时生效，
   改 env 不会改已有库——只改 env 的结果是"配置比库新，服务全挂"。
3. `docker compose up -d` 让容器拿到新 env。注意 `restart` **不重读** `.env`，
   必须 `up -d`（会 recreate）。

三个坑：

- **最容易漏的是"半迁移"**。本轮第一版把 `NEO4J_AUTH`（容器初始化用）换成了新口令，
  却漏了 backend 的 `NEO4J_PASSWORD=` 那一行。结果是 Neo4j 认新口令、后端拿旧口令，
  图谱鉴权失败——而 `/health` 依然返回 `ok`（它只看进程），`/chat/stream` 也照样 200，
  只是答案全走纯向量。判据不能只看健康检查：要跑一条真实图谱题，或
  `MATCH (n) RETURN count(n)` 拿到节点数（实测 742）。教训是——**替换清单按"变量名"穷举，
  不要按"服务"穷举**：同一个口令在 compose 里出现两次（容器侧 + 客户端侧），名字还不一样。

- 脚本里不要写 `${user}` 这类 Bash 变量引用。`set -u` 下它会直接退出，
  一条 ALTER 都不执行，而 `.env` 已经换了新口令——上一步就是这么翻车的。
  用 Python 传值，口令走环境变量透传，别进 argv（`ps` 可见）。
- `backup.sh` 是从容器 env 读 `MYSQL_ROOT_PASSWORD` 的，所以 recreate 之后它自动跟随新口令，
  不需要单独改。但**如果只 ALTER 库没 recreate 容器**，备份会拿旧口令失败。


### 5.7 清缓存端点：为什么必须有（2026-11-05 晚补）

`POST /admin/cache/clear`，鉴权与其它写入端点一致（无 key 401）。

两个用途，都不是"顺手加个接口"：

**一、测方差必须先清缓存。** 门禁跑 `--repeat 3` 时，如果不清缓存，
第 2、3 轮全部命中缓存、拿到同一份答案，极差恒为 0。
**那个"稳定"是缓存造成的，不是模型的。** 这种假结论比没有结论更危险，
因为它在报告上看起来像是"我们很稳定"。

**二、线上改检索策略后不必重启。** 重启 backend 要重载约 1.2 GiB 模型，40 秒起步；
清缓存是毫秒级，且不影响正在进行的请求。

顺带把口径钉死：**清缓存后测出的极差 0，只能说明"可复现"，不能说明"鲁棒"。**
前者是"同一批问题重复跑结果一致"，后者是"换个问法也能答对"——后者要靠评测集覆盖面，
不是靠重复跑。两者别混。

**踩到的坑（下次别再踩）：** 端点一开始插在 `/health` 后面，而 `require_api_key`
定义在第 190 行。装饰器在**导入期**就求值，于是 `NameError: name 'require_api_key' is not defined`，
表现为容器反复重启：`docker compose ps` 里 backend 的 Up 时间永远在几十秒内徘徊
（先是 "Up 3 seconds"，三分钟后再看变成 "Up 20 seconds"）。

判据：`docker compose logs backend --tail 20 | grep NameError`。
结论：**FastAPI 路由必须写在它依赖的函数定义之后**，新端点统一加到文件后部，
不要图方便往 `/health` 旁边塞。


---

## 6. 怎么复现这一轮的结论

```bash
# 两层测试：宿主机跑纯逻辑，容器内跑需要后端依赖的
bash /root/rag_deploy/run_tests.sh

# 备份 + 恢复验证
bash /root/rag_deploy/backup.sh && bash /root/rag_deploy/restore_verify.sh

# 评测门禁（约 4 分钟）
python3 /root/rag_deploy/eval_gate.py --min-score 0.90

# 注入探针（5 条，其中 2 条走"有资料"分支）
python3 /root/rag_deploy/injection_probe.py

# 降级演练：停 Neo4j 后必须用没问过的题（否则命中缓存，结论是假的）
docker compose stop neo4j
python3 /root/rag_deploy/drill_degrade.py down3 test_cases3.json C003,C004
docker compose start neo4j

# 启动依赖演练：Neo4j 停着重启后端，看能不能起来
docker compose stop neo4j && docker compose restart backend
curl -s http://127.0.0.1:8000/health
docker compose start neo4j
```


---

## 7. MCP 接入（2026-11-05）

把知识库暴露成 MCP 工具，外部 agent（Claude Desktop / Cursor / 自研客户端）
可以直接检索和问答。完整接入指南见 **MCP.md**，这里只记运维要点。

### 7.1 架构：薄客户端，不在本地加载模型

```
MCP 客户端 ──stdio/JSON-RPC──> mcp_server.py ──HTTP+X-API-Key──> backend :8000
                                                              ├── POST /search
                                                              └── POST /chat/stream
```

**必须这样分层。** backend 容器已经常驻 bge 向量模型 + reranker 约 1.2G，
MCP 进程如果自己 import 一遍 `rag_agent`，会再加载一份模型——3.6G 内存直接打爆，
而且两份模型的缓存还不一致。MCP 层只负责协议翻译，一行检索逻辑都没重复实现。

### 7.2 依赖：`pip --target`，不是 venv

这台机器的 `python3-venv` 没装（`ensurepip` 缺失），`python3 -m venv` 直接失败。
所以用 `--target` 装到独立目录，靠 `PYTHONPATH` 隔离：

```bash
cd /root/rag_deploy
pip3 install --target .mcp_libs mcp httpx     # 约 43MB，已加入 .gitignore

# 验证
PYTHONPATH=.mcp_libs python3 -c "from mcp.server.mcpserver import MCPServer; print('ok')"
```

⚠️ **mcp 2.x 把 `FastMCP` 改名成 `MCPServer` 了**：
`from mcp.server.mcpserver import MCPServer`。
网上（以及 2.x 之前）的教程全是 `from mcp.server.fastmcp import FastMCP`，
在这个版本上会抛 `ModuleNotFoundError` 并提示迁移。
安装的版本是 **2.3.0**。

### 7.3 两个工具，不是五个

| 工具 | 作用 | 典型耗时 |
|---|---|---|
| `search_knowledge_base(query, top_k)` | 只返回证据片段，不生成 | 2.5–3.5s（图谱路径）/ 10s+（无实体全量） |
| `ask_knowledge_base(question, session_id)` | 完整问答，返回答案 + 来源 + 相关度 | 3–10s |

**刻意不扩充。** 工具越多模型选错概率越高，越需要"循环探索"兜底；
这两个动词的边界很清楚：要证据还是要素材、快还是全。

### 7.4 验证方式：协议握手，不是单测

单测只能证明"函数逻辑对"，证明不了"它是个 MCP Server"。
`mcp_smoke.py` 起子进程、发 JSON-RPC 帧、按 id 收响应：

```bash
cd /root/rag_deploy && PYTHONPATH=.mcp_libs python3 mcp_smoke.py
# ✅ MCP 协议握手全部通过
```

覆盖 11 项：initialize / capabilities / tools/list / 参数 schema /
两个工具的真实调用 / **stdout 未被污染**（往 stdout 里 print 一行就能搞坏协议）。

### 7.5 两个坑（都踩过，记下来）

1. **客户端读 stdout 不能用 `select()`。**
   `Popen(text=True)` 在 Python 层做缓冲，`select()` 看的是 OS fd——
   数据已进 Python 缓冲区，OS 侧就是"没数据"。表现为"服务端明明回了，
   客户端永远等"。改成后台线程 + `queue`。排查花了很久，
   因为 `python3 mcp_server.py < init.json` 一直是正常的。

2. **改完模块要重启容器。**
   上传 `evidence.py` 后没重启 backend，模块还是旧的，于是 MCP 返回里
   残留 `【图谱检索结果】【来源：test.txt】` 标签。
   判据：看返回文本里有没有本该被剥掉的标签。**热更新对 Python 模块不生效。**

### 7.6 已画出的边界

- **SSE 模式没有鉴权**。MCP 客户端对自定义 header 的支持不统一，硬塞 token 会让
  很多客户端连不上。所以默认只监听 `127.0.0.1`，跨机使用请走 SSH 隧道：
  `ssh -N -L 8765:127.0.0.1:8765 root@<host>`。
  **不要把 8765 开到公网**——它等于一个免鉴权的知识库读取接口。
- **无实体过滤的检索要 10s+**（未命中尤其慢，因为要跑完多跳才判定落空）。
  MCP 客户端默认超时可能不够，`RAG_TIMEOUT` 默认给到 180s。

---

## 8. 检索路由规则（2026-11-06）

### 8.1 它解决什么

每次检索原本都要发一次 DeepSeek 判"是不是关系题"。实测这一步单独 **1130ms**
（日志 08:51:47.369 → 08:51:48.499），占总耗时 14%。改成"规则先判明显的，
判不了才问模型"后，线上 **29% 的检索**（34 次里 10 次）不再发这次请求。

### 8.2 怎么加一条规则

改 `retrieval_router.py` 的 `RULES`，但**必须先离线比对，不能直接上**：

```bash
# 用 48 道真实题库比对"规则判断"与"模型判断"
docker compose exec -T backend python verify_router.py > /tmp/router.json
python3 -c "
import json; d=json.load(open('/tmp/router.json'))
print('精确率', d['precision'], '误判', d['misjudged'])
"
```

判定标准只有一条：**`misjudged` 必须为空**。

- `MISJUDGE`（规则说关系、模型说不是）→ 危险，会白跑一次图谱（1723ms）+ 一次降级
- `miss_safe`（规则没命中、模型说是）→ 安全，只是回到原路径，不花钱不出错

第一版就撞了一个：`workplace` 规则里写了 "做什么工作"，把 C004
「小陈是做什么工作的？」判成关系题——**职业是属性，不是工作单位**。
删掉后 15/15 精确率 100%。**这类误判评测分数看不出来**（答案最终一样，只是慢），
只能靠离线比对和单测拦。

### 8.3 三条不能破的约束

1. **只在恰好一个实体时启用**。没有实体时"哪个省"不成立，而"谁养了猫"这类
   聚合题恰恰要交给模型判断（`test_no_entity_disables_rules` 守着）。
2. **只认句式，不认单词**。"宠物"裸词不算，必须"养了什么宠物"这种带谓词的形状。
3. **判不了一律回退模型**，不许猜。漏判零代价，误判倒赔 600ms 以上。

### 8.4 怎么确认它没改变行为

```bash
docker compose logs backend --since 15m | grep -E "路由规则命中|规则未命中" | wc -l
python3 eval_gate.py --cases test_cases2.json --repeat 1   # 加规则时 0.9474；补地理层级后 1.0000
```

日志里 `⚡ 路由规则命中 rule:xxx，跳过模型意图判断` 表示省了一次往返；
`⚡ 规则未命中（llm/no_ent），回退模型判断` 表示走了原路径。**两者都要能看见**，
否则没法证明"省了"和"回退"各自发生了多少次。

---

## 9. 地理层级：城市 → 省（2026-11-06）

### 9.1 为什么会有这一节

`graph_retriever` 的 Cypher 提示词里，规则 2 写着「涉及省份必须走两跳
`人物-[:住在]->城市-[:属于]->省份`」，示例 2 举的例子正是「小陈住在哪个省」。

**但图上从来没有这条边。** 抽取器只能连文本里共现的词：原文写「我住在南京」，
就只有 `(小智)-[:住在]->(南京)`，没有"江苏"二字，所以省这一跳永远返回 0 行。

症状很迷惑：Cypher 生成得完全正确、检索置信度 0.9389、答案却说"资料里没说明"。
**看起来像模型不肯推断，实际是数据没满足查询层的契约。**

### 9.2 怎么补的

```bash
docker compose exec -T backend python backfill_geo.py            # dry-run，先看计划
docker compose exec -T backend python backfill_geo.py --apply    # 执行
docker compose exec -T backend python backfill_geo.py --audit    # 按来源统计属于边
docker compose exec -T backend python backfill_geo.py --rollback # 回滚
```

映射表在 `geo_hierarchy.py`（60+ 城市，只对图里存在的节点生效）。
执行结果：**743 节点 / 893 边（+1/+18）**。第二次执行 0 新增，幂等。

### 9.3 三个必须知道的约束

**① 补出来的边带 `origin='geo_backfill'`，抽取的没有这个属性。**
这是唯一的区分手段，也是回滚的依据：

```cypher
MATCH ()-[r:属于]->() WHERE r.origin = 'geo_backfill' DELETE r
```

`--audit` 能把两类分开数（当前：抽取 206 条 / 补边 18 条）。
**不要删这个标记**，删了就分不清哪些边是原文有的、哪些是补的。

**② `__Entity__` 节点上不能加任何自定义属性。**

两个都踩过：

- `source='geo_hierarchy'` → 会出现在 `sources` 里，等于给用户一个**假出处**
- `created_by='geo_backfill'` → 会被 `_collect_entities` 渲染成实体属性，
  答案里真的出现过「小李住在浙江省。**（补充：浙江是 geo_backfill。）**」

原因：`_collect_entities` 把节点所有非 `id`/`source` 的属性都当成"要展示给用户的属性"。
审计信息只能放边上或外部报告里。

**③ 改映射表后必须跑单测。**

```bash
python3 -m pytest tests/test_geo_hierarchy.py -q   # 29 条，0.06s
```

守三条不变量：映射目标必须是合法省名、省名不能出现在城市位（防方向写反）、
自环只能给直辖市。**方向写反是最危险的错**——`(江苏)-[:属于]->(南京)` 会让
"住在哪个省"答出一个城市名，而且不报错。

### 9.4 已知边界

- **补出来的边没有原文出处。** 正确表述是「城市名有出处，省市从属关系来自结构补全」。
  这不是缺陷，是结构补全的固有属性；但对外要能主动说清，不能等人问。
- **直辖市用自环** `(上海)-[:属于]->(上海)`。它们的省级单位就是自己，
  不建自环则"住在哪个省"对北京/上海/重庆必然查不到。
- **两跳查询必须同时返回中间节点。** 只 `RETURN properties(province)` 会让
  `sources` 指向省节点所在的文档（比如菜谱"江苏扬州"），而不是人物资料。
  提示词规则 2 已加这条约束。

### 9.5 怎么确认它生效且无副作用

```bash
# 应各自答出省名，且 sources 含人物自己的文档
docker compose exec -T backend python probe_ask.py "小智住在哪个省？" "小陈住在哪个省？"
# 评测应从 0.9474 变为 1.0000，且拒答类仍然 4/4
python3 eval_gate.py --cases test_cases2.json --repeat 1
```

**判断标准不是"分数涨了"，而是"拒答题没掉"。** 如果拒答题开始乱答，
说明改动扩散到了生成层的口径，而不只是补了数据 —— 那是另一个性质的问题。

---

## 10. 品种层级：品种 → 类别（2026-11-07）

### 10.1 为什么会有这一节

对抗集 C009「养猫的是哪几个人」在图谱那一步返回 0 行。生成的 Cypher 是：

```cypher
MATCH (pet:宠物) WHERE pet.type = "猫" RETURN properties(pet)
```

两个错叠在一起：

1. **`type` 只在"豆豆"一个节点上有**（抽取阶段的不一致：其余宠物都走 `品种是` 关系）。
   它在 Schema 里露了脸，模型就当成通用字段用了 —— 命中率恒为 0，还不报错。
2. **提示词没有聚合示例**：四条示例全是"某个人 + 某条关系"，没有"问谁、问题里却没有人名"的形状。

### 10.2 为什么不能只修提示词

`品种是` 只到品种这一层：

```
代码 -[:品种是]-> 橘猫        拿铁 -[:品种是]-> 猫
```

`WHERE 品种="猫"` 只命中拿铁，**漏掉小李**。漏人的名单比检索落空更危险：
它会理直气壮地给出不完整的答案。所以必须补一层 `品种 -[:属于类别]-> 类别`。

### 10.3 三个必须知道的约束

**① 不要复用 `属于`。** 它在图谱里被用得极宽（小刘属于校篮球队、片儿川属于杭州、
珍珠奶茶属于奶茶），`graph_retriever.py` 顶部专门写了"绝对不能当类型用"。
再往里塞层级只会更乱。所以用独立关系 `属于类别`，语义单一，也便于整体回滚。

**② 查询必须写成 `-[:属于类别*0..1]->`。** `*0..1` 里的 **0 跳**覆盖
"品种本身就是类别"（拿铁的品种就是"猫"），**1 跳**覆盖"品种属于类别"（橘猫属于猫）。
少了变长匹配，两类宠物只能覆盖一类 —— 而且不报错，只是少人。

**③ 新建的类别节点不能带任何自定义属性。** `狗` 节点只设标签和 id。
踩过：给节点设 `source` 会变成假出处，设 `created_by` 会被渲染进答案
（「浙江是 geo_backfill」）。审计信息只能放边上（`r.origin='breed_backfill'`）或报告文件里。

### 10.4 怎么补、怎么回滚

```bash
docker compose exec -T backend python backfill_breed.py                    # dry-run，默认
docker compose exec -T backend python backfill_breed.py --apply --clean-type
docker compose exec -T backend python backfill_breed.py --audit
docker compose exec -T backend python backfill_breed.py --rollback         # 边、节点、被清的属性一起还原
```

`--clean-type` 清掉宠物上那个只出现一次的 `type` 属性（品种信息在 `品种是` 边里已有一份）。
不幂等的地方只有这一处，所以它单独开关、单独记录到 `/root/logs/breed_backfill.json`，
回滚时按报告还原。

### 10.5 怎么确认它生效且无副作用

```bash
# 模板能渲染（新规则里混进单个花括号时 .format 会抛 KeyError，py_compile 查不出来）
docker compose exec -T backend python probe_graph.py "养猫的是哪几个人？"
# 应返回 小李/代码（橘猫）+ 小吴/拿铁（猫），来源 test5+test10
```

对照检查清单：

| 检查 | 为什么 |
|---|---|
| 养猫的返回 **2 个人** | 只返回 1 个说明 `*0..1` 没生效或层级没补上 |
| 养狗的返回 小陈/豆豆 | 验证新建的 `狗` 节点可用 |
| C005「小陈养的狗是什么品种」仍对 | 删 `type` 后品种知识靠 `品种是` 归类补全，这条是回归哨兵 |
| 4 道拒答题仍 `conf=None` | 拒答口径没被生成层的改动带偏 |
| `test_cases2` 三轮 1.0000 且逐题翻转 0 | 提示词是全局的，改示例可能影响别的题型 |

### 10.6 已知边界

- **图谱通了不等于线上会走它。** 这次 C009 线上实际是向量答对的（`route=hybrid`），
  因为 `is_relation_query` 对措辞敏感：原话判"是关系题"，模型改写的
  "养猫的人有哪些"判"否"。图谱这条路**修好了但没被走到**。
  要不要让聚合题优先走图谱，动的是 `retrieval_router.py` 里"聚合题交给模型"那条
  约束（C004 误判换来的），属于独立决策，不要顺手改。
- **品种映射表是常识表，不是从语料抽的。** 覆盖 51 条常见中宠物叫法；
  语料出现表外品种时会漏，表现是"少人"而不是"报错"。
- **只补两端都在图里的边。** 品种节点不存在就不补（`plan_edges` 的不变量），
  否则会凭空造节点、污染节点数审计。

