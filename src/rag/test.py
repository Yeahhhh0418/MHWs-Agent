import sys
sys.path.insert(0, "src")

from rag.mainrag import EquipmentCraftingRAG
from rag.tokenizer import chinese_char_tokenizer

r = EquipmentCraftingRAG().setup()

# ①②③ 各准备几条：明显相关的 / 边缘的 / 完全不相关的
QUERIES = {
    "相关":   ["豪鬼套怎么制作", "锁刃龙头需要什么材料"],
    "边缘":   ["猎人装备推荐", "有什么好用的套装"],
    "不相关": ["今天天气怎么样", "帮我写一段 python 代码"],
}

for label, qs in QUERIES.items():
    for q in qs:
        print(f"\n[{label}] {q}")

        # ── 向量路：真实相关性分数（[0,1]，越大越相关）
        vec = r._vectorstore.similarity_search_with_relevance_scores(q, k=r._vector_k)
        vec_scores = [round(s, 4) for _, s in vec]
        print(f"  向量分数 top{len(vec_scores)}: {vec_scores}")

        # ── BM25 路：原始分数（量纲不同于向量，不可直接比较）
        bm25 = r._retriever.retrievers[0]
        raw = bm25.vectorizer.get_scores(chinese_char_tokenizer(q))
        top = sorted(raw, reverse=True)[:r._bm25_k]
        print(f"  BM25 分数 top{len(top)}: {[round(float(x), 3) for x in top]}")
