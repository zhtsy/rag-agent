import jieba
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List

OLLAMA_HOST = "http://127.0.0.1:11434"
PERSIST_DIR = "./chroma_db"

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

    def __init__(self, documents, metadatas, k=10):
        super().__init__(
            documents=documents,
            metadatas=metadatas,
            k=k
        )
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.k]
        return [
            Document(page_content=self.documents[i], metadata=self.metadatas[i])
            for i in top_indices if scores[i] > 0
        ]


def build_bm25_retriever():
    embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    data = vectorstore.get()
    return SimpleBM25Retriever(
        documents=data["documents"],
        metadatas=data["metadatas"],
        k=10
    )


bm25_retriever = build_bm25_retriever()