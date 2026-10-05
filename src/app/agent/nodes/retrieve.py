"""
src/app/agent/nodes/retrieve.py —— 检索节点：按当前库和关键词做混合检索。
"""
from app.agent.state import AgentState, RAGResult
from app.logger import get_logger
from app.rag import mainrag

logger = get_logger("nodes.retrieve")


# ── RAG 实例懒加载缓存 ───────────────────────────────────
_rag_cache = {}


def _get_rag(db: str):
    """按库名取 RAG 实例（BaseRAG）；首次会 setup() 并缓存，之后直接复用。"""
    if db in _rag_cache:
        return _rag_cache[db]
    logger.info("首次加载数据库", extra={"db": db})
    if db == "equipment":
        rag = mainrag.EquipmentCraftingRAG()
    elif db == "weapon":
        rag = mainrag.WeaponCraftingRAG()
    elif db == "material":
        rag = mainrag.MaterialInfoRAG()
    elif db == "monster":
        rag = mainrag.MonsterInfoRAG()
    else:
        raise ValueError(f"未知数据库: {db}")
    rag.setup()
    _rag_cache[db] = rag
    return rag


def retrieve_node(state: AgentState) -> dict:
    db = state["current_db"]
    keyword = state["query_keyword"]
    iteration = state.get("iteration_count", 0)

    try:
        rag = _get_rag(db)
        docs = rag.retrieve(keyword)

        parts = []
        doc_ids = []
        for doc in docs:
            parts.append(doc.page_content)
            doc_ids.append(doc.metadata["_doc_id"])

        content = "\n\n".join(parts)
    except Exception as e:
        logger.error("检索失败，使用占位内容", extra={"db": db, "error": str(e)})
        content = f"（数据库 {db} 检索失败，暂时跳过）"
        doc_ids = []

    new_result = RAGResult(db=db, keyword=keyword, content=content,doc_ids=doc_ids)
    return {
        "collected_info": [new_result],
        "iteration_count": iteration + 1,
    }
