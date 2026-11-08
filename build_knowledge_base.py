import os
import shutil
from collections import Counter

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from embeddings_shared import get_embeddings

KNOWLEDGE_DIR = "./knowledge_base"
PERSIST_DIR = "./chroma_db"

# 与线上服务共用同一个单例，避免模型定义分散成多份后漂移
embeddings = get_embeddings()


# ========== 实体映射表：文件名 → 实体名 ==========
entity_map = {
    # ===== 人物档案 =====
    "test.txt": "小智",
    "test1.txt": "小明",
    "test2.txt": "小王",
    "test3.txt": "小赵",
    "test4.txt": "小赵",
    "test5.txt": "小李",
    "test6.txt": "小张",
    "test7.txt": "小刘",
    "test8.txt": "小陈",
    "test9.txt": "小周",
    "test10.txt": "小吴",
    # ===== 游戏 =====
    "lol_jialiao.txt": "加里奥",
    "Terraria.txt": "泰拉瑞亚",
    # ===== 美食 =====
    "chuancai.txt": "川菜",
    "yuecai.txt": "粤菜",
    "kaoya.txt": "北京烤鸭",
    "dongbeimeishi.txt": "东北美食",
    "jiangzhemeishi.txt": "江浙美食",
    "xibeimeishi.txt": "西北美食",
    # ===== 菜谱 =====
    "fanqiechaodan.txt": "番茄炒蛋",
    "qingjiaochaodan.txt": "青椒炒蛋",
    "hupijidan.txt": "虎皮鸡蛋",
}


def load_documents():
    all_docs = []
    for root, dirs, files in os.walk(KNOWLEDGE_DIR):
        for file in files:
            path = os.path.join(root, file)
            try:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                elif file.endswith(".txt") or file.endswith(".md"):
                    loader = TextLoader(path, encoding="utf-8")
                else:
                    continue

                docs = loader.load()
                # ========== 打实体标签 ==========
                filename = os.path.basename(path)
                if filename in entity_map:
                    for doc in docs:
                        doc.metadata["entity"] = entity_map[filename]
                        doc.metadata["source_file"] = filename

                all_docs.extend(docs)
                print(f"📄 加载: {path} (实体: {entity_map.get(filename, '无')})")
            except Exception as e:
                print(f"❌ 加载 {path} 失败: {e}")
    return all_docs


def backfill_entity_name(chunks):
    """把主体名回填到每个块开头。

    人物档案多为第一人称书写，自我介绍里只出现一次名字，
    分块后后续块（如"我养了一只金毛叫豆豆"）就丢失了主体，
    导致按实体过滤、BM25 关键词都检索不到。回填后每块自带主体锚点。
    """
    filled = 0
    for c in chunks:
        ent = c.metadata.get("entity")
        if not ent:
            continue
        prefix = f"（{ent}）"
        if c.page_content.startswith(prefix):
            continue
        c.page_content = prefix + c.page_content
        filled += 1
    return filled


def safe_clear_dir(path):
    """清空目录内容但保留目录本身。

    chroma_db 在 compose 中是独立挂载点，直接 rmtree 目录会在 rmdir 时报 EBUSY，
    且此时内容已被删除，处于半损坏状态。因此只清内容。
    """
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        p = os.path.join(path, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        except Exception as e:
            print(f"⚠️ 清理 {p} 失败: {e}")
    print(f"🗑️ 已清空旧向量库内容: {path}")


docs = load_documents()
print(f"✅ 共加载 {len(docs)} 个文档")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)
chunks = text_splitter.split_documents(docs)
print(f"✅ 共切分 {len(chunks)} 个文本块（原始）")

filled = backfill_entity_name(chunks)
print(f"✅ 已回填主体名到 {filled} 个块")

# 删除旧库，避免重复
safe_clear_dir(PERSIST_DIR)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)
print(f"🎉 知识库构建完成！共 {len(chunks)} 个块已存入 {PERSIST_DIR}")

# ========== 构建后自检 ==========
cnt = Counter(c.metadata.get("entity", "无") for c in chunks)
print("📊 各实体块数:")
for ent, n in sorted(cnt.items(), key=lambda x: -x[1]):
    print(f"   {ent:8s} {n}")

sample = [c for c in chunks if c.metadata.get("entity") == "小陈"]
print(f"🔍 小陈 块数={len(sample)}，含'豆豆'的块数={sum('豆豆' in c.page_content for c in sample)}")
for i, c in enumerate(sample):
    print(f"   块{i}: {c.page_content[:50]}...")
