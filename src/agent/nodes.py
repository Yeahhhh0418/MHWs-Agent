"""
agent/nodes.py —— 干净版节点：路由 / 检索 / 分析 / 生成（无追踪、无重试、无评估）
"""

import json

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agent.state import AgentState, RAGResult
from models.chatllm import get_llm
from prompts.prompt import ROUTE_PROMPT, ANALYZE_PROMPT, GENERATE_PROMPT
from mhwi_agent.config import get_config
from mhwi_agent.logger import get_logger
from rag import mainrag

logger = get_logger("nodes")


def _invoke(system_prompt: str, user_content: str) -> str:
    """单次 LLM 调用，返回文本。"""
    llm = get_llm()
    return llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]).content


def _parse_json(text: str, fallback: dict) -> dict:
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else fallback
    except (json.JSONDecodeError, TypeError):
        return fallback


# ── 路由节点 ──────────────────────────────────────────────
def route_node(state: AgentState) -> dict:
    user_msg = state["messages"][-1].content  #这里起步的时候是个什么值？
    logger.info("开始路由", extra={"question": user_msg[:100]})

    try:
        result = _parse_json(_invoke(ROUTE_PROMPT, user_msg), {})
        first_db = result["first_db"]
        first_keyword = result["first_keyword"]
    except Exception as e:
        logger.error("路由失败，使用默认路由", extra={"error": str(e)})
        first_db, first_keyword = "equipment", user_msg

    return {
        "user_question": user_msg,
        "current_db": first_db,
        "query_keyword": first_keyword,
        "iteration_count": 0,
        "collected_info": [],
    }


# ── RAG 实例懒加载缓存 ───────────────────────────────────
_rag_cache = {}


def _get_rag(db: str):
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


# ── 检索节点 ──────────────────────────────────────────────
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


def format_collected_info(collected_info: list) -> str:
    if not collected_info:
        return "未检索到任何相关信息。"
    formatted_text = "【已检索到的参考资料】：\n"
    for i, result in enumerate(collected_info, 1):
        formatted_text += f"{i}. 数据库: {result['db']}\n"
        formatted_text += f"   关键词: {result['keyword']}\n"
        formatted_text += f"   内容: {result['content']}\n\n"
    return formatted_text


# ── 分析节点 ──────────────────────────────────────────────
def analyze_node(state) -> dict:
    collected_text = format_collected_info(state["collected_info"])

    try:
        result = _parse_json(_invoke(ANALYZE_PROMPT, f"""
原始问题：{state["user_question"]}
已检索轮次：{state["iteration_count"]}
已收集信息：
{collected_text}

请判断是否需要继续检索，输出 JSON。
"""), {})
        next_action = result["next_action"]
    except Exception as e:
        logger.error("分析失败，强制 finish", extra={"error": str(e)})
        next_action = "finish"

    if next_action == "continue":
        return {
            "next_action": "continue",
            "current_db": result.get("next_db", "material"), # material为默认值
            "query_keyword": result.get("next_keyword", state["user_question"]),
        }
    else:
        return {"next_action": "finish"}


# ── 生成节点 ──────────────────────────────────────────────
def generate_node(state: AgentState) -> dict:
    collected_text = format_collected_info(state["collected_info"])

    try:
        answer = _invoke(GENERATE_PROMPT, f"""
玩家问题：{state["user_question"]}

以下是从数据库检索到的相关信息：
{collected_text}

请根据以上信息，用自然、友好的语气回答玩家的问题。
""")
    except Exception as e:
        logger.error("生成失败，返回原始检索内容", extra={"error": str(e)})
        answer = collected_text

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ── 条件边函数 ────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    max_iterations = get_config().get("graph", {}).get("max_iterations", 5)
    if state["iteration_count"] >= max_iterations:
        return "finish"
    if state["next_action"] == "finish":
        return "finish"
    return "continue"
