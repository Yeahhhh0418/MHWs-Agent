"""
src/app/agent/nodes/route.py —— 路由节点：判断问题核心实体，决定第一个查哪个库。
"""

import json

from app.agent.nodes.common import invoke
from app.agent.state import AgentState
from app.logger import get_logger, new_request_id
from app.prompts import ROUTE_PROMPT

logger = get_logger("nodes.route")

# 路由认不出问题该查哪个库时的兜底库。
# 原来 route 和 analyze 各写各的（"equipment" / "material"），互相打架（项目完善清单 §2.2）。
DEFAULT_DB = "equipment"


def _parse_json(text: str, fallback: dict) -> dict:
    """宽松解析：解析不出来、或结果不是 JSON 对象时返回 fallback（dict），不抛异常。"""
    try:
        result = json.loads(text)
        # 解析出来不是对象（比如模型返回了个数组）也算失败
        return result if isinstance(result, dict) else fallback
    except (json.JSONDecodeError, TypeError):
        # 交给调用方给的兜底值，不往上抛 —— 路由失败不该让整轮问答中断
        return fallback


db_list = ["equipment", "weapon", "material", "monster"]


def route_node(state: AgentState) -> dict:
    # 图的起点只塞了一条 HumanMessage，就是用户原话 —— route 是第一个节点，
    # generate 之前没人再往 messages 里加东西，所以这里取到的没被加工过
    user_msg = state["messages"][-1].content
    # 这次问答的编号：后面每个节点的日志都挂着它
    rid = new_request_id()
    logger.info("开始路由", extra={"question": user_msg[:100]})

    try:
        # 问模型：该先查哪个库、用什么词
        result = _parse_json(invoke(ROUTE_PROMPT, user_msg), {})
        first_db = result["first_db"]
        # 模型可能编出一个不存在的库，退回默认
        if first_db not in db_list:
            logger.warning(
                "路由给出了不存在的库，回退默认",
                extra={"first_db": first_db, "fallback": DEFAULT_DB},
            )
            first_db = DEFAULT_DB
        # 关键词会原样拿去检索，所以要尽量完整 —— 括号、序号、"♀" 都属于名字的一部分
        first_keyword = result["first_keyword"]
        logger.info("路由完成", extra={"db": first_db, "keyword": first_keyword})
    except Exception as e:  # noqa: BLE001 —— 路由失败就回退默认库，流程继续往下走。
        logger.error("路由失败，使用默认路由", extra={"error": str(e)})
        first_db, first_keyword = DEFAULT_DB, user_msg

    # 初始化这一轮的起点状态：编号、原问题、第一个库和关键词
    return {
        "request_id": rid,
        "user_question": user_msg,
        "current_db": first_db,
        "query_keyword": first_keyword,
        "iteration_count": 0,
        "collected_info": [],
    }
