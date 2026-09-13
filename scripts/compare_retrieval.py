"""
agent/scripts/compare_retrieval.py —— 检查装备混合检索效果（含命中率）

对每条查询，打印命中的前 3 条文档，并自动判断是否命中"期望装备名"，
最后输出一个可量化的命中率，用来对比调参前后的效果。

用法（在仓库根目录 C:\\code\\MHWI_agent 下）：
    python -m agent.scripts.compare_retrieval
"""

from agent.rag.equipment_rag import EquipmentCraftingRAG


TOP_K = 3  # 只看前几条，命中率也按前几条算（Hit@3）


# 每条查询 -> 期望命中的装备名（子串匹配即可）。
# expected 为 None 表示模糊查询（没有唯一标准答案），只打印、不计入命中率。
QUERIES = [
    ("炎尾龙头盔怎么做", ["炎尾龙头盔"]),   # 全名
    ("炎尾龙的头",       ["炎尾龙头盔"]),   # 俗称：头 → 头部防具 → 头盔
    ("缠蛙的衣服",       ["缠蛙铠甲"]),     # 俗称：衣服 → 胸部防具 → 铠甲
    ("辟兽的手",         ["辟兽腕甲"]),     # 俗称：手 → 腕部防具 → 腕甲
    ("沙海龙的腰",       ["沙海龙腰甲"]),   # 俗称：腰 → 腰部防具 → 腰甲
    ("希望面具",         ["希望面具"]),     # 完整名直接匹配
    ("能回血的装备",     None),             # 模糊查询，不计分
    ("加攻击力的头盔",   None),             # 模糊查询，不计分
]


def _doc_name(doc) -> str:
    """取文档的正式装备名（metadata.name）。"""
    return (doc.metadata or {}).get("name") or ""


def _is_hit(docs, expected) -> bool:
    """前几条里是否有一条的名称包含任意一个期望装备名。"""
    for doc in docs:
        name = _doc_name(doc)
        if any(exp in name for exp in expected):
            return True
    return False


def main():
    rag = EquipmentCraftingRAG()
    rag.setup()

    scored = 0
    hit = 0

    for query, expected in QUERIES:
        docs = rag.retrieve(query)  # 只检索，不生成

        print(f"\n查询：{query}")
        for i, doc in enumerate(docs[:TOP_K], 1):
            name = _doc_name(doc)
            preview = doc.page_content[:40].replace("\n", " ")
            print(f"  {i}. [{name}] {preview}")

        if expected is None:
            print("  （模糊查询，不计分）")
            continue

        scored += 1
        if _is_hit(docs[:TOP_K], expected):
            hit += 1
            print("  -> 命中")
        else:
            print("  -> 未命中（期望含：" + " / ".join(expected) + "）")

    print("\n================ 命中率 ================")
    if scored:
        print(f"  可评分查询 {scored} 条，命中 {hit} 条")
        print(f"  命中率 = {hit}/{scored} = {hit / scored:.0%}")
    else:
        print("  无可评分查询")


if __name__ == "__main__":
    main()
