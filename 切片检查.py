from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("./knowledge_base/test6.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = splitter.split_documents(docs)

for i, chunk in enumerate(chunks):
    print(f"=== 块 {i+1} ===")
    print(chunk.page_content)
    print(f"包含'小张': {'小张' in chunk.page_content}")
    print(f"包含'甜点': {'甜点' in chunk.page_content}")
    print()