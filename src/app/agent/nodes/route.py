"""
src/app/agent/nodes/route.py —— 路由节点：判断问题核心实体，决定第一个查哪个库。
"""

import time

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from app.agent.state import DB, AgentState
from app.llm import invoke_with_fallback
from app.logger import get_logger, new_request_id
from app.prompts import ROUTE_PROMPT

logger = get_logger("nodes.route")

# 路由认不出问题该查哪个库时的兜底库。
# 原来 route 和 analyze 各写各的（"equipment" / "material"），互相打架（项目完善清单 §2.2）。
DEFAULT_DB = "equipment"

# 拼错格式、或偶发故障时，最多再给一次机会
ROUTE_MAX_ATTEMPTS = 2


class RouteError(Exception):
    """
    [WHY] 路由输出不合格要和别的故障分得开 —— 只有这一类才值得回喂重试
          （网络故障重问一遍没用）。跟 analyze 的 AnalyzeError 是同一个道理。
    [FIX] 异常消息里写清原因，route_node 把它原样回喂给模型重试一次。
    """


class RouteDecision(BaseModel):
    """
    [WHY] 原来是手写 json.loads + 手动取字段（那个 _parse_json）。两个毛病：
          ① 库名是自由字符串 —— 模型能编出第五个库，只能事后手工查一遍；
          ② 模型只要在 JSON 后面多写一句解释，整段就作废 —— 实测 30 次里崩了 5 次，
             连对照组的题都会崩（概率性的，跟题目模不模糊无关）。
             崩了以后走兜底：默认库 + **整句问题**当关键词，实测捞回 4 件完全无关的防具。
    [FIX] 跟 analyze 用同一套：schema 约束字段和库名；LangChain 那边底层的解析器能容忍
          "JSON 后面多一段解释"和"被 ``` 包起来"（json.loads 两种都扛不住）。
          不合格当场抛错，由 route_node 回喂重试一次。
    """

    first_db: DB = Field(description="第一个要查的库")
    # ⚠️ min_length=1 是必须的：str 允许空串，而空串拿去检索毫无意义（实测捞回 10 条随机东西）
    first_keyword: str = Field(min_length=1, description="最核心的检索关键词，会原样拿去检索")


def _invoke_and_parse_route(user_msg: str, retry_hint: str = "") -> RouteDecision:
    """调一次模型并校验（RouteDecision）；不合格抛 RouteError，消息供上层回喂重试。"""
    content = user_msg
    # 重试时把上次为什么不合格一并喂回去，不然模型会原样再错一遍
    if retry_hint:
        content += f"\n\n你上一次的输出无法使用，原因：{retry_hint}\n请按约定的字段重新输出。"

    messages = [SystemMessage(content=ROUTE_PROMPT), HumanMessage(content=content)]

    def call(llm: BaseChatModel) -> RouteDecision:
        """
        [WHY] 跟 analyze 一样必须显式写 method="json_mode"：DeepSeek 的思考模式模型
              不支持 tool calling，报 "Thinking mode does not support this tool_choice"，
              而 with_structured_output 默认走的正是那条路 —— 主模型会直接 400。
        """
        bound = llm.with_structured_output(RouteDecision, method="json_mode")
        try:
            return bound.invoke(messages)
        except (ValidationError, OutputParserException) as e:
            # 输出格式不合格是提示词/字段约定的问题，换模型也一样不合格 —— 该走回喂重试
            raise RouteError(f"模型输出不符合约定字段：{e}") from e

    # RouteError 不换模型：格式问题换谁都一样，该走上层的回喂重试
    decision, _used = invoke_with_fallback(call, no_fallback_on=(RouteError,))
    return decision


def route_node(state: AgentState) -> dict:
    # 图的起点只塞了一条 HumanMessage，就是用户原话 —— route 是第一个节点，
    # generate 之前没人再往 messages 里加东西，所以这里取到的没被加工过
    user_msg = state["messages"][-1].content
    # 这次问答的编号：后面每个节点的日志都挂着它
    rid = new_request_id()
    logger.info("开始路由", extra={"question": user_msg[:100]})

    # 重试状态：记着上一次为什么没成
    decision = None
    last_error = ""
    for attempt in range(ROUTE_MAX_ATTEMPTS):
        try:
            decision = _invoke_and_parse_route(user_msg, retry_hint=last_error)
            # 成了就跳出，不再试第二次
            break
        except RouteError as e:
            # 格式类失败：记下原因，下一轮拼进提示词回喂
            last_error = str(e)
            logger.warning("路由输出不合格", extra={"attempt": attempt + 1, "error": last_error})
        except Exception as e:  # noqa: BLE001 —— 偶发故障也重来一次，仍失败才降级并留痕
            last_error = f"调用失败：{type(e).__name__}: {e}"
            logger.warning("路由调用失败", extra={"attempt": attempt + 1, "error": last_error})

    # 初始化这一轮的起点状态：编号、原问题、第一个库和关键词
    result = {
        "request_id": rid,
        "user_question": user_msg,
        "iteration_count": 0,
        "collected_info": [],
        # route 是第一个节点，这里记下就约等于"整个问答的起点"
        "started_at": time.monotonic(),
    }

    if decision is None:
        # 两次都没成 —— 降级：退回默认库，拿原问题当关键词（总比不查强）。
        # 必须留痕：不留的话这次失败在评测里一个字都看不见，只能翻日志
        degrade_reason = f"路由两次都没成：{last_error}"
        result["current_db"] = DEFAULT_DB
        result["query_keyword"] = user_msg
        result["degraded"] = True
        result["degrade_reason"] = degrade_reason
        logger.error("路由降级，退回默认库", extra={"db": DEFAULT_DB, "reason": degrade_reason})
    else:
        # 关键词会原样拿去检索，所以要尽量完整 —— 括号、序号、"♀" 都属于名字的一部分
        result["current_db"] = decision.first_db
        result["query_keyword"] = decision.first_keyword
        result["degraded"] = False
        logger.info("路由完成", extra={"db": decision.first_db, "keyword": decision.first_keyword})

    return result
