import os
import jieba
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List

from embeddings_shared import get_embeddings
from kb_paths import resolve as resolve_chroma_dir

# 与 rag_agent 同一口径的绝对路径：相对路径在别的 cwd 下会静默建空库。
# 也必须与 rag_agent 读**同一个目录**——两处分别写死的话，切索引时漏改一处
# 就会出现"向量查 A 语料、BM25 查 B 语料"的混合结果，且不报任何错。
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR, KB_DIR_NOTE = resolve_chroma_dir(_BASE_DIR)

ENTITY_TERMS = [
    "小智", "小明", "小王", "小赵", "小李", "小张", "小刘", "小陈", "小周", "小吴",
    "加里奥", "泰拉瑞亚", "虎皮鸡蛋", "番茄炒蛋", "青椒炒蛋",
    "北京烤鸭", "川菜", "粤菜", "东北美食", "江浙美食", "西北美食"
]

for term in ENTITY_TERMS:
    jieba.add_word(term)


class SimpleBM25Retriever(BaseRetriever):
    """继承 BaseRetriever，兼容 LangChain 的 EnsembleRetriever"""

    documents: List[str]
    metadatas: List[dict]
    k: int = 10
    bm25: object = None
    tokenized_corpus: list = None
    # 只在这几个来源文件里检索（空 = 全库）。
    #
    # 这是"法名硬过滤"在 BM25 这一侧的落点。向量侧可以用 chroma 的
    # `where={"source_file": {"$in": [...]}}`，但 BM25 是内存全量索引，
    # 没有 filter 接口 —— 于是它的 top-10 名额可能被别的法占满，
    # 向量侧收窄了也没用（EnsembleRetriever 会把这 10 块照单全收）。
    #
    # 实现方式是在打分后掩码，不是重建索引：31,856 块的 BM25Okapi 构建要几十秒，
    # 每次查询重建不现实。掩码只是多一次 O(n) 遍历，微秒级。
    source_files: List[str] = []

    def __init__(self, documents, metadatas, k=10, source_files=None):
        super().__init__(
            documents=documents,
            metadatas=metadatas,
            k=k,
            source_files=list(source_files or []),
        )
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)

        allow = set(self.source_files) if self.source_files else None
        if allow is None:
            pool = range(len(scores))
        else:
            pool = [
                i for i in range(len(scores))
                if (self.metadatas[i] or {}).get("source_file") in allow
            ]

        top_indices = sorted(pool, key=lambda i: scores[i], reverse=True)[: self.k]
        return [
            Document(page_content=self.documents[i], metadata=self.metadatas[i])
            for i in top_indices if scores[i] > 0
        ]

    def scoped(self, source_files, k=None) -> "SimpleBM25Retriever":
        """同一份索引、限定来源文件的**浅拷贝**（不重建 BM25）。

        必须拷贝而不是改自己：`bm25_retriever` 是模块级单例，
        直接改 `source_files` 会在并发请求之间互相污染 —— 用户 A 问了
        《劳动合同法》、用户 B 同时问了《专利法》，两人会看到对方的作用域。

        `model_copy` 不走 `__init__`，所以 `tokenized_corpus` / `bm25`
        是共享引用，只多出一个 list。代价是一次浅拷贝。
        """
        upd = {"source_files": list(source_files or [])}
        if k:
            upd["k"] = int(k)
        try:
            return self.model_copy(update=upd)      # pydantic v2
        except AttributeError:
            return self.copy(update=upd)            # pydantic v1


def build_bm25_retriever():
    # 这里的 Chroma 只用来 get() 出全量文本块喂给 BM25，
    # embedding_function 在 get() 时根本不会被调用，纯粹是构造函数必填参数。
    # 传共享单例（与 rag_agent 用的是同一个模型对象），不再新加载一份 400MB。
    embedding_model = get_embeddings()
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    data = vectorstore.get()
    return SimpleBM25Retriever(
        documents=data["documents"],
        metadatas=data["metadatas"],
        k=10
    )


bm25_retriever = build_bm25_retriever()