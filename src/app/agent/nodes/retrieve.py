"""
src/app/agent/nodes/retrieve.py —— 检索节点：按当前库和关键词做混合检索。
"""

from app.agent.state import AgentState, RAGResult
from app.logger import bind_request_id, get_logger
from app.rag import mainrag
from app.rag.base import BaseRAG

logger = get_logger("nodes.retrieve")


# ── RAG 实例懒加载缓存 ───────────────────────────────────
_rag_cache = {}


def _get_rag(db: str) -> BaseRAG:
    """按库名取 RAG 实例（BaseRAG）；首次会 setup() 并缓存，之后直接复用。"""
    # 建过一次就一直复用 —— setup() 要加载向量模型，很贵
    if db in _rag_cache:
        return _rag_cache[db]
    logger.info("首次加载数据库", extra={"db": db})
    # 按库名挑实现
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
    # setup 里干三件事：读数据、编编号、建检索器
    rag.setup()
    _rag_cache[db] = rag
    return rag


def retrieve_node(state: AgentState) -> dict:
    # 节点入口重绑编号：LangGraph 给每个节点的上下文各是一份，不会自动传过来
    bind_request_id(state.get("request_id", ""))
    # 这一轮查哪个库、用什么词，都是 route / analyze 事先定好的
    db = state["current_db"]
    keyword = state["query_keyword"]
    iteration = state.get("iteration_count", 0)

    rag = _get_rag(db)
    # 默认没降级，下面出错时改写
    degraded = False
    degrade_reason = ""
    try:
        docs = rag.retrieve(keyword)

        # 拆成两份：正文给模型读，编号给评测核对命中
        parts = []
        doc_ids = []
        for doc in docs:
            parts.append(doc.page_content)
            doc_ids.append(doc.metadata["_doc_id"])

        content = "\n\n".join(parts)
        # 一条都没查到也得给句话 —— 不然模型分不清"这个词没找到"和"资料本来就是空的"
        if not content:
            content = f"（数据库 {db} 没有找到与关键词 {keyword} 相关的资料）"

        # 成功也打一条：不然多轮检索在控制台完全看不出走了几轮、每轮查了什么
        logger.info("检索完成", extra={"db": db, "keyword": keyword, "hits": len(docs)})

    except Exception as e:  # noqa: BLE001 —— 降级但写进 state.degraded，评测看得见，不是静默吞掉
        logger.error("检索失败，使用占位内容", extra={"db": db, "error": str(e)})
        content = f"（数据库 {db} 这次没能取到资料）"
        doc_ids = []
        degraded = True
        degrade_reason = f"检索 {db} 失败：{type(e).__name__}"

    # 打包成一条检索结果，会被追加进 collected_info
    new_result = RAGResult(db=db, keyword=keyword, content=content, doc_ids=doc_ids)
    return {
        "collected_info": [new_result],
        "iteration_count": iteration + 1,
        "degraded": degraded,
        "degrade_reason": degrade_reason,
    }
