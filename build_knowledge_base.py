import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

KNOWLEDGE_DIR = "./knowledge_base"
PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"

embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url="http://127.0.0.1:11434")

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
    "fanqiechaodan.txt":"番茄炒蛋",
    "qingjiaochaodan.txt":"青椒炒蛋",
    "hupijidan.txt":"虎皮鸡蛋",
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

                all_docs.extend(docs)
                print(f"📄 加载: {path} (实体: {entity_map.get(filename, '无')})")
            except Exception as e:
                print(f"❌ 加载 {path} 失败: {e}")
    return all_docs

docs = load_documents()
print(f"✅ 共加载 {len(docs)} 个文档")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = text_splitter.split_documents(docs)
print(f"✅ 共切分 {len(chunks)} 个文本块")

# 删除旧库，避免重复
import shutil
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print(f"🗑️ 已删除旧向量库")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)
print(f"🎉 知识库构建完成！共 {len(chunks)} 个块已存入 {PERSIST_DIR}")