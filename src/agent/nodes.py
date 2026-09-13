"""
agent/nodes.py —— LangGraph 的四个节点 + 条件边函数（加固 + 追踪版）
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agent.state import AgentState, RAGResult
from agent.chatllm import llm
from agent.prompts.prompt import ROUTE_PROMPT, ANALYZE_PROMPT, GENERATE_PROMPT
from agent.config import get_config
from agent.logger import get_logger
from agent.retry import safe_json_parse, retry_call
from agent.tracker import track_node
from agent.evaluator import check_empty_answer, check_named_entity_overlap
from agent.rag import EquipmentCraftingRAG, ItemDataRAG, MonsterDataRAG

logger = get_logger("nodes")


# ── 路由节点 ──────────────────────────────────────────────
def route_node(state: AgentState) -> dict:
    return track_node("route", _route_impl, state)


def _route_impl(state: AgentState) -> dict:
    user_msg = state["messages"][-1].content
    logger.info("开始路由", extra={"question": user_msg[:100]})

    def _call_llm():
        return llm.invoke([
            SystemMessage(content=ROUTE_PROMPT),
            HumanMessage(content=user_msg),
        ])

    try:
        response = retry_call(_call_llm, max_retries=2)
        result = safe_json_parse(
            response.content,
            required_keys=["first_db", "first_keyword"],
            fallback={"first_db": "equipment", "first_keyword": user_msg},
        )
    except Exception as e:
        logger.error("路由 LLM 彻底失败，使用默认路由", extra={"error": str(e)})
        result = {"first_db": "equipment", "first_keyword": user_msg}

    return {
        "user_question": user_msg,
        "current_db": result["first_db"],
        "query_keyword": result["first_keyword"],
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
        rag = EquipmentCraftingRAG()
    elif db == "item":
        rag = ItemDataRAG()
    elif db == "monster":
        rag = MonsterDataRAG()
    else:
        raise ValueError(f"未知数据库: {db}")
    rag.setup()
    _rag_cache[db] = rag
    return rag


# ── 检索节点 ──────────────────────────────────────────────
def retrieve_node(state: AgentState) -> dict:
    return track_node("retrieve", _retrieve_impl, state)


def _retrieve_impl(state: AgentState) -> dict:
    db = state["current_db"]
    keyword = state["query_keyword"]
    iteration = state.get("iteration_count", 0)

    try:
        rag = _get_rag(db)
        content = rag.query(keyword)
    except Exception as e:
        logger.error("检索失败，使用占位内容", extra={"db": db, "error": str(e)})
        content = f"（数据库 {db} 检索失败，暂时跳过）"

    new_result = RAGResult(db=db, keyword=keyword, content=content)
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
    return track_node("analyze", _analyze_impl, state)


def _analyze_impl(state) -> dict:
    collected_text = format_collected_info(state["collected_info"])

    def _call_llm():
        return llm.invoke([
            SystemMessage(content=ANALYZE_PROMPT),
            HumanMessage(content=f"""
原始问题：{state["user_question"]}
已检索轮次：{state["iteration_count"]}
已收集信息：
{collected_text}

请判断是否需要继续检索，输出 JSON。
"""),
        ])

    try:
        response = retry_call(_call_llm, max_retries=2)
        result = safe_json_parse(
            response.content,
            required_keys=["next_action"],
            fallback={"next_action": "finish"},
        )
    except Exception as e:
        logger.error("分析 LLM 失败，强制 finish", extra={"error": str(e)})
        result = {"next_action": "finish"}

    if result["next_action"] == "continue":
        return {
            "next_action": "continue",
            "current_db": result.get("next_db", "item"),
            "query_keyword": result.get("next_keyword", state["user_question"]),
        }
    else:
        return {"next_action": "finish"}


# ── 生成节点 ──────────────────────────────────────────────
def generate_node(state: AgentState) -> dict:
    return track_node("generate", _generate_impl, state)


def _generate_impl(state: AgentState) -> dict:
    collected_text = format_collected_info(state["collected_info"])

    def _call_llm():
        return llm.invoke([
            SystemMessage(content=GENERATE_PROMPT),
            HumanMessage(content=f"""
玩家问题：{state["user_question"]}

以下是从数据库检索到的相关信息：
{collected_text}

请根据以上信息，用自然、友好的语气回答玩家的问题。
"""),
        ])

    try:
        response = retry_call(_call_llm, max_retries=2)
        answer = response.content
    except Exception as e:
        logger.error("生成 LLM 失败，返回原始检索内容", extra={"error": str(e)})
        answer = collected_text

    # grounding guard：回答里的【实体】能否在检索内容中找到
    if not check_empty_answer(answer):
        retrieved_texts = [r["content"] for r in state["collected_info"]]
        coverage = check_named_entity_overlap(answer, retrieved_texts)
        logger.info("回答实体覆盖率", extra={"coverage": round(coverage, 2)})
        if coverage < 0.5:
            answer += "\n\n（提示：以上部分信息可能未在资料库中核实，请以游戏内实际为准。）"

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
