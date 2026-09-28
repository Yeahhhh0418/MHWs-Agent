"""看一眼怪物检索是谁在捣乱：BM25、向量、最终返回，分开打印排名。"""
import sys

sys.path.insert(0, "src")

from rag.mainrag import MonsterInfoRAG

QUERIES = [
    "护火龙的生态",
    "微型欧米茄是什么怪物",
    "火龙这个怪要怎么打？",
    "火龙和雌火龙哪个好打",
    "天空王者是谁",
]

rag = MonsterInfoRAG().setup()
bm25 = rag._retriever.retrievers[0]
vector = rag._retriever.retrievers[1]


def show(title, docs, n=3):
    print("-- %s --" % title)
    if not docs:
        print("   （空）")
    for i, d in enumerate(docs[:n], 1):
        print("   %d. %s" % (i, d.metadata["_doc_id"]))


for q in QUERIES:
    print("=" * 46)
    print("问:", q)
    print("   认出的名字:", rag._names_in_question(q))
    show("BM25", bm25.invoke(q))
    show("向量", vector.invoke(q))
    show("最终返回", rag.retrieve(q), 6)
    print()
