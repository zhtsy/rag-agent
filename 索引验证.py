from rag_agent import vectorstore

results = vectorstore.get()
print(f"总块数: {len(results['ids'])}")
sources = set()
for meta in results['metadatas']:
    sources.add(meta.get('source'))
for s in sorted(sources):
    print(s)