"""
src/app/agent/nodes/analyze.py —— 分析节点：判断还缺什么、要不要继续查，以及它的条件边。

结构化输出卡住字段和库名；决策过程记进 state.decision_trace；模型输出不合格会回喂重试一次。
"""
from functools import lru_cache
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError

from app.agent.nodes.common import format_collected_info
from app.agent.state import AgentState
from app.config import get_config
from app.llm import get_llm
from app.logger import get_logger
from app.prompts import ANALYZE_PROMPT

logger = get_logger("nodes.analyze")

ANALYZE_MAX_ATTEMPTS = 2
MAX_SAME_QUERY_REPEAT = 2   # 同一组 (库, 关键词) 最多允许查几次，见 _count_searched
DB = Literal["equipment", "weapon", "material", "monster"]


class AnalyzeError(Exception):
    """
    [WHY] analyze 输出不合格要和别的故障分得开 —— 只有这一类才值得重试（网络故障重问一遍没用）。
          原来一律 except Exception，两类混在一起，只能悄悄降级成 finish。
    [FIX] 异常消息里写清原因，analyze_node 把它原样回喂给模型重试一次。
    """


class AnalyzeDecision(BaseModel):
    """
    [WHY] 原来靠提示词写「请输出 JSON」+ json.loads，库名是自由字符串（能编出第五个库），
          字段缺了只能等 KeyError。
    [FIX] 不合格当场报错，由 analyze_node 回喂给模型重试。
    """
    next_action: Literal["continue", "finish"]
    reason: str = Field(description="一句话说清：为什么继续，或为什么可以停了")
    next_db: DB | None = Field(default=None, description="next_action 是 continue 时必填")
    next_keyword: str | None = Field(default=None, description="next_action 是 continue 时必填")


@lru_cache(maxsize=1)
def _decision_llm() -> Runnable:
    """绑了 AnalyzeDecision 的模型（只绑一次）；单独拎出来是为了测试时能整个换掉。"""
    return get_llm().with_structured_output(AnalyzeDecision)


def _check_decision(decision: AnalyzeDecision) -> None:
    """
    [WHY] 「说 continue 就必须给出去哪查、查什么」这条规则，Pydantic 的 schema 表达不了 ——
          Optional 字段在两种 next_action 下都合法。
    """
    if decision.next_action == "continue":
        if not decision.next_db:
            raise AnalyzeError("next_action 是 continue，但没有给出 next_db")
        if not decision.next_keyword:
            raise AnalyzeError("next_action 是 continue，但没有给出 next_keyword")


def _invoke_and_parse_analyze(user_content: str, retry_hint: str = "") -> AnalyzeDecision:
    """调一次模型并校验（AnalyzeDecision）；不合格抛 AnalyzeError，消息供上层回喂重试。"""
    content = user_content
    if retry_hint:
        content += (
            f"\n\n你上一次的输出无法使用，原因：{retry_hint}\n"
            "请按约定的字段重新输出。"
        )

    try:
        decision = _decision_llm().invoke([
            SystemMessage(content=ANALYZE_PROMPT),
            HumanMessage(content=content),
        ])
    except (ValidationError, OutputParserException) as e:
        raise AnalyzeError(f"模型输出不符合约定字段：{e}") from e

    _check_decision(decision)
    return decision


def _count_searched(state: AgentState, db: str, keyword: str) -> int:
    """
    [WHY] 同一组 (库, 关键词) 再查一遍，拿到的结果和上一遍一字不差 —— 第三遍就是纯绕圈了。
          但不能一票否决模型：它可能真有理由再确认一次，而且"查过了"的口径是我们定的，
          未必和它一致（§2.5）。
    [FIX] 换关键词查同一个库不受这条限制；全局还有 graph.max_iterations 兜底。
    """
    n = 0
    for item in state["collected_info"]:
        if item["db"] == db and item["keyword"] == keyword:
            n += 1
    return n


def _decide(state: AgentState, source: str, action: str, db: str | None, keyword: str | None,
            reason: str, degraded: bool = False) -> dict:
    """把一次决策打包成节点返回值；同时写日志和 decision_trace（§2.4 §2.8）。"""
    entry = {
        "iteration": state["iteration_count"],
        "source": source,          # llm / code / degrade —— 这个决定是谁下的
        "action": action,
        "db": db,
        "keyword": keyword,
        "reason": reason,
    }
    logger.info("analyze 决策", extra=entry)

    result = {
        "next_action": action,
        "degraded": degraded,
        "degrade_reason": reason if degraded else "",
        "decision_trace": [entry],
    }
    if action == "continue":
        result["current_db"] = db
        result["query_keyword"] = keyword
    return result


def analyze_node(state: AgentState) -> dict:
    collected_text = format_collected_info(state["collected_info"])
    user_content = f"""
原始问题：{state["user_question"]}
已检索轮次：{state["iteration_count"]}
已收集信息：
{collected_text}
"""

    decision = None
    last_error = ""
    for attempt in range(ANALYZE_MAX_ATTEMPTS):
        try:
            decision = _invoke_and_parse_analyze(user_content, retry_hint=last_error)
            break
        except AnalyzeError as e:
            last_error = str(e)
            logger.warning("analyze 输出不合格", extra={"attempt": attempt + 1, "error": last_error})
        except Exception as e: # noqa: BLE001 —— 网络/未知故障：重问一遍没用，直接降级（但要留痕，不能装作答完了）
            last_error = f"调用失败（不重试）：{type(e).__name__}: {e}"
            logger.error("analyze 调用失败", extra={"error": last_error})
            break

    if decision is None:
        return _decide(state, "degrade", "finish", None, None, last_error, degraded=True)

    if decision.next_action == "continue":
        times = _count_searched(state, decision.next_db, decision.next_keyword)
        if times >= MAX_SAME_QUERY_REPEAT:
            return _decide(state, "code", "finish", None, None,
                           f"「{decision.next_db} / {decision.next_keyword}」已经查过 {times} 次，再来一遍不会多出信息")

    return _decide(state, "llm", decision.next_action, decision.next_db,
                   decision.next_keyword, decision.reason)


# ── 条件边函数 ────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    """analyze 之后走哪条边（str）：轮次到顶或决定停就走 finish，否则 continue。"""
    max_iterations = get_config().get("graph", {}).get("max_iterations", 5)
    if state["iteration_count"] >= max_iterations:
        logger.info("analyze 轮次到顶，强制 finish", extra={"iteration": state["iteration_count"]})
        return "finish"
    if state["next_action"] == "finish":
        return "finish"
    return "continue"
