"""
知识图谱构建 v2（不依赖 langchain_experimental）

为什么要重写：
  1. 容器里没装 langchain_experimental，原版 LLMGraphTransformer 直接 ImportError，
     这就是图谱从 10 月之后一直没更新的原因。
  2. 原版只从 chroma 读块，却顺手加载了 SentenceTransformer（~400MB），
     本机只有 3G 内存，没必要。这里用 chromadb 直读，不加载任何本地模型。
  3. 原版关系类型由模型自由生成，容易出现同义词漂移（拥有/有/持有）。
     这里改成白名单枚举，抽取结果必须落在枚举内。

关键改进：分块文本已回填主体名，形如「（小张）我养了一只金毛…」。
提示词里显式要求把「我」解析成括号里的主体名，这是 `养了` 关系能补全的前提。

用法：
  python build_graph.py                # 增量抽取 + 增量写入（MERGE，不删旧数据）
  python build_graph.py --limit 3      # 只抽 3 块，用于验证
  python build_graph.py --reuse        # 只写缓存，不调 API
  python build_graph.py --reset        # 写入前清空图谱（危险，需配合备份）
"""

import os
import re
import json
import time
import argparse
from collections import defaultdict

import chromadb
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage
from neo4j import GraphDatabase

load_dotenv()

CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
COLLECTION = os.environ.get("CHROMA_COLLECTION", "langchain")
CACHE_PATH = os.environ.get("GRAPH_CACHE", "./graph_extract_cache.jsonl")

# 连接参数统一从 settings.py 取，这里不再各写一份默认值。
# 历史教训：口令原本在 4 个文件里各写一遍，换口令时漏掉 build_graph.py，
# 报出来的却是 "AuthenticationRateLimit"，查了半小时才找到是漏改。
from settings import neo4j_kwargs  # noqa: E402

_neo4j = neo4j_kwargs()
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD = _neo4j["url"], _neo4j["username"], _neo4j["password"]

# 枚举约束：抽取结果必须落在这里，否则丢弃。
# 比原版多了 音乐/物品/学科 —— 原版没有这三类，"球鞋"会被塞成食物、"数学"塞成游戏。
ALLOWED_NODES = ["人物", "地点", "游戏", "食物", "宠物", "职业", "菜谱", "组织",
                 "音乐", "物品", "学科"]
ALLOWED_RELS = [
    "喜欢", "不喜欢", "住在", "养了", "擅长", "属于",
    "制作", "由...制成", "拥有", "工作于",
]
# 同义关系归一化：模型偶尔换个说法，映射回枚举
REL_ALIASES = {
    "拥有": ["有", "持有", "具备"],
    "喜欢": ["喜爱", "爱好"],
    "不喜欢": ["讨厌", "厌恶"],
    "擅长": ["善于", "精通"],
    "养了": ["养", "饲养", "养着"],
    "住在": ["居住于", "生活在"],
    "工作于": ["任职于", "供职于"],
}

SYSTEM_PROMPT = """你是知识图谱抽取器。从给定文本中抽取实体与关系，输出 json。

【实体类型】只能是：人物、地点、游戏、食物、宠物、职业、菜谱、组织、音乐、物品、学科
  - 宠物：猫、狗等动物的名字（如「豆豆」「笨笨」「拿铁」）
  - 音乐：乐队、歌手、曲风（如「摇滚乐」「枪花」「周杰伦」）
  - 物品：球鞋、手机、车等具体物件
  - 学科：数学、物理、计算机等
【关系类型】只能是：喜欢、不喜欢、住在、养了、擅长、属于、制作、由...制成、拥有、工作于

【重要规则】
1. 文本开头的「（某某）」是这段话的叙述主体。文中出现的「我」「我的」一律指代该主体，
   必须解析成真实姓名，不要输出「我」。
2. 关系方向一律从「人物/主体」指向「被描述的对象」。
   例：「我养了一只金毛叫豆豆」→ 主体 -[养了]-> 豆豆
3. 实体 id 用文本中的原词，不要加修饰、不要自造。人物用姓名（如「小陈」），
   宠物用宠物名（如「豆豆」），不要写成「小陈的宠物」。
4. 只抽取文本明确陈述的事实，不要推理、不要补充常识。
5. 关系类型必须从上面枚举里挑，不要自创近义词。
6. 宁可漏，不要错：实体若实在无法归类，就不要输出；类型拿不准时选最贴近的一个。
7. 【空间词不是从属关系】「附近 / 旁边 / 楼下 / 对面 / 隔壁 / 门口」只表示位置接近，
   绝不能据此生成「属于」或「住在」。「我住在医院附近的一个小区」
   只能得到「主体 -[住在]-> 小区」，不允许出现「小区 -[属于]-> 医院」。
8. 【住在只接行政区】「住在」的目标只能是城市/省份/国家这类行政区。
   小区、公寓、宿舍、房子等具体居所不要连到「住在」上，也不要从它们往城市/机构连「属于」。
   「工作于」「任职于」只接单位，不要和「住在」混用。

【输出格式】严格输出 json，不要 markdown 代码块：
{"entities":[{"id":"小陈","type":"人物"},{"id":"豆豆","type":"宠物"}],
 "relations":[{"source":"小陈","relation":"养了","target":"豆豆"}]}
没有可抽取内容时输出 {"entities":[],"relations":[]}"""


def log(msg):
    print(msg, flush=True)


def load_chunks(limit=None):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(COLLECTION)
    data = col.get(include=["documents", "metadatas"])
    docs = data["documents"] or []
    metas = data["metadatas"] or [{} for _ in docs]
    items = list(zip(docs, metas))
    if limit:
        items = items[:limit]
    return items


def load_cache():
    done = {}
    if not os.path.exists(CACHE_PATH):
        return done
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["idx"]] = rec
            except Exception:
                continue
    return done


# 「住在医院附近的一个小区」曾被抽成 小区 -[属于]-> 医院，
# 结果"小陈住在哪个省"两跳查出"医院"，答案直接跑偏。
# 提示词里已经加了约束，这里再做一道词表兜底——模型不听话时至少不污染图。
_RESIDENCE_WORDS = ("小区", "公寓", "宿舍", "住宅", "房子", "老家")


def _is_spatial_misread(rel, source, target):
    """拦掉「空间邻近」被误读成「从属/居住」的关系。"""
    if rel == "属于" and any(w in source for w in _RESIDENCE_WORDS):
        return True
    if rel == "住在":
        # 住在只接行政区：具体居所或机构（医院/公司/学校）都不算居住地
        if any(w in target for w in _RESIDENCE_WORDS + ("医院", "公司", "学校")):
            return True
    return False


def normalize(data, subject):
    """把模型输出收敛到枚举内，并解析「我」。"""
    ents = []
    seen = set()
    for e in data.get("entities") or []:
        if not isinstance(e, dict):
            continue
        eid = str(e.get("id") or "").strip()
        etype = str(e.get("type") or "").strip()
        if not eid or etype not in ALLOWED_NODES:
            continue
        if eid in ("我", "自己", "本人") and subject:
            eid = subject
        if (eid, etype) in seen:
            continue
        seen.add((eid, etype))
        ents.append({"id": eid, "type": etype})

    ent_type = {e["id"]: e["type"] for e in ents}
    rels = []
    seen_r = set()
    for r in data.get("relations") or []:
        if not isinstance(r, dict):
            continue
        s = str(r.get("source") or "").strip()
        t = str(r.get("target") or "").strip()
        rt = str(r.get("relation") or "").strip()
        if s in ("我", "自己", "本人") and subject:
            s = subject
        if t in ("我", "自己", "本人") and subject:
            t = subject
        if not s or not t or s == t:
            continue
        for canon, aliases in REL_ALIASES.items():
            if rt in aliases:
                rt = canon
                break
        if rt not in ALLOWED_RELS:
            continue
        if s not in ent_type or t not in ent_type:
            continue
        if _is_spatial_misread(rt, s, t):
            continue
        key = (s, rt, t)
        if key in seen_r:
            continue
        seen_r.add(key)
        rels.append({"source": s, "rel": rt, "target": t,
                     "stype": ent_type[s], "ttype": ent_type[t]})
    return {"entities": ents, "relations": rels}


def extract_one(llm, text, subject, retries=3):
    msgs = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=text)]
    last_err = None
    for attempt in range(retries):
        try:
            resp = llm.invoke(msgs)
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                raise ValueError("no json in response")
            data = json.loads(m.group(0))
            return normalize(data, subject)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"抽取失败: {last_err}")


def write_graph(rels, chunk_idx):
    if not rels:
        return 0
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    grouped = defaultdict(list)
    for r in rels:
        grouped[(r["stype"], r["rel"], r["ttype"])].append(r)

    written = 0
    with driver.session() as s:
        for (stype, rel, ttype), items in grouped.items():
            # 标签/关系类型都来自白名单，拼接安全
            assert stype in ALLOWED_NODES and ttype in ALLOWED_NODES and rel in ALLOWED_RELS
            # 先按 id 在 __Entity__ 上 MERGE，再 SET 具体标签。
            # 不能写 MERGE (s:人物 {id})：若同 id 的老节点带的是别的标签，
            # 会匹配不上而新建一个重复节点（同一 id 两个标签），后续 MATCH 会返回重复行。
            q = (
                f"UNWIND $rows AS row "
                f"MERGE (s:`__Entity__` {{id: row.source}}) "
                f"SET s:`{stype}` "
                f"MERGE (t:`__Entity__` {{id: row.target}}) "
                f"SET t:`{ttype}` "
                f"MERGE (s)-[r:`{rel}`]->(t) "
                f"SET r.chunk = $chunk "
                f"RETURN count(r) AS c"
            )
            rows = [{"source": r["source"], "target": r["target"]} for r in items]
            rec = s.run(q, rows=rows, chunk=chunk_idx).single()
            written += rec["c"] if rec else 0
    driver.close()
    return written


def reset_graph():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as s:
        before = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        s.run("MATCH (n) DETACH DELETE n")
        log(f"🧹 已清空图谱（原有 {before} 个节点）")
    driver.close()


def stats():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as s:
        n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        r = s.run("MATCH ()-[x]->() RETURN count(x) AS c").single()["c"]
        log(f"📊 节点={n} 关系={r}")
        for rec in s.run("MATCH ()-[x]->() RETURN type(x) AS t, count(*) AS c ORDER BY c DESC"):
            log(f"   {rec['t']}: {rec['c']}")
        pets = list(s.run("MATCH (a:人物)-[r:养了]->(b) RETURN a.id AS a, b.id AS b"))
        log(f"养了关系 {len(pets)} 条")
    driver.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse", action="store_true", help="只写缓存，不调 API")
    ap.add_argument("--reset", action="store_true", help="写入前清空图谱")
    ap.add_argument("--write-only", action="store_true", help="不抽取不写入，仅统计")
    ap.add_argument("--extract-only", action="store_true", help="只抽取到缓存，不写库")
    args = ap.parse_args()

    chunks = load_chunks(limit=args.limit)
    log(f"待处理分块: {len(chunks)}")

    cache = load_cache()
    log(f"已有缓存: {len(cache)}")

    if not args.reuse and not args.write_only:
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0,
                           model_kwargs={"response_format": {"type": "json_object"}})
        t0 = time.time()
        with open(CACHE_PATH, "a", encoding="utf-8") as f:
            for i, (text, meta) in enumerate(chunks):
                if i in cache:
                    continue
                subject = (meta or {}).get("entity") or ""
                try:
                    out = extract_one(llm, text, subject)
                except Exception as e:
                    log(f"[{i}] FAIL {e}")
                    continue
                rec = {"idx": i, "entity": subject,
                       "source_file": (meta or {}).get("source_file", ""),
                       "entities": out["entities"], "relations": out["relations"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                cache[i] = rec
                if (i + 1) % 5 == 0 or i == len(chunks) - 1:
                    el = time.time() - t0
                    log(f"进度 {i+1}/{len(chunks)} 用时 {el:.0f}s 本块关系 {len(rec['relations'])}")
        log(f"抽取完成，缓存 {len(cache)} 条")
    else:
        log("跳过抽取，使用缓存")

    if args.write_only:
        stats()
        return

    if args.extract_only:
        # 中文经 workbench 回显会坏，这里只打 ASCII 键的计数，真实内容走文件下载核对
        from collections import Counter
        c = Counter()
        pet = 0
        for rec in cache.values():
            for r in rec["relations"]:
                c[r["rel"]] += 1
                if r["rel"] == "养了":
                    pet += 1
        log(f"CACHED_CHUNKS={len(cache)}")
        log(f"TOTAL_RELS={sum(c.values())}")
        log(f"YANGLE={pet}")
        for i, name in enumerate(ALLOWED_RELS):
            log(f"REL_{i}={c.get(name, 0)}")
        return

    if args.reset:
        reset_graph()

    # 按 chunk 顺序写入，便于断点复跑
    all_rels = 0
    for idx in sorted(cache):
        all_rels += write_graph(cache[idx]["relations"], idx)
    log(f"写入完成，关系写入 {all_rels} 次")

    stats()


if __name__ == "__main__":
    main()
