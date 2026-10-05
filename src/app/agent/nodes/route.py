"""
src/app/agent/nodes/route.py —— 路由节点：判断问题核心实体，决定第一个查哪个库。
"""
import json

from app.agent.nodes.common import invoke
from app.agent.state import AgentState
from app.logger import get_logger
from app.prompts import ROUTE_PROMPT

logger = get_logger("nodes.route")

# 路由认不出问题该查哪个库时的兜底库。
# 原来 route 和 analyze 各写各的（"equipment" / "material"），互相打架（项目完善清单 §2.2）。
DEFAULT_DB = "equipment"


def _parse_json(text: str, fallback: dict) -> dict:
    """宽松解析：解析不出来、或结果不是 JSON 对象时返回 fallback（dict），不抛异常。"""
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else fallback
    except (json.JSONDecodeError, TypeError):
        return fallback


def route_node(state: AgentState) -> dict:
    user_msg = state["messages"][-1].content  #这里起步的时候是个什么值？
    logger.info("开始路由", extra={"question": user_msg[:100]})

    try:
        result = _parse_json(invoke(ROUTE_PROMPT, user_msg), {})
        first_db = result["first_db"]
        first_keyword = result["first_keyword"]
    except Exception as e:
        logger.error("路由失败，使用默认路由", extra={"error": str(e)})
        first_db, first_keyword = DEFAULT_DB, user_msg

    return {
        "user_question": user_msg,
        "current_db": first_db,
        "query_keyword": first_keyword,
        "iteration_count": 0,
        "collected_info": [],
    }
