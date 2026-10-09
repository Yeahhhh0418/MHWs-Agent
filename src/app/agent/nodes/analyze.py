"""
src/app/agent/nodes/analyze.py —— 分析节点：判断还缺什么、要不要继续查，以及它的条件边。

结构化输出卡住字段和库名；决策过程记进 state.decision_trace；模型输出不合格会回喂重试一次。
"""

from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from app.agent.nodes.common import format_collected_info
from app.agent.state import AgentState
from app.config import get_config
from app.llm import invoke_with_fallback
from app.logger import bind_request_id, get_logger
from app.prompts import ANALYZE_PROMPT

logger = get_logger("nodes.analyze")

# 输出不合格、或偶发故障时，最多再给一次机会
ANALYZE_MAX_ATTEMPTS = 2
# 库名卡死在字面量里 —— 模型编不出第五个库
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
    # reason 每次都要 —— 不管停还是继续，都得说得出来为什么
    reason: str = Field(description="一句话说清：为什么继续，或为什么可以停了")
    # 后两个只在 continue 时有意义。"continue 就必须给值"这条 schema 管不了，见 _check_decision
    next_db: DB | None = Field(default=None, description="next_action 是 continue 时必填")
    next_keyword: str | None = Field(default=None, description="next_action 是 continue 时必填")


def _check_decision(decision: AnalyzeDecision) -> None:
    """
    [WHY] 「说 continue 就必须给出去哪查、查什么」这条规则，Pydantic 的 schema 表达不了 ——
          Optional 字段在两种 next_action 下都合法。
    """
    # 说了要继续，就必须交代"去哪查、查什么"
    if decision.next_action == "continue":
        if not decision.next_db:
            raise AnalyzeError("next_action 是 continue，但没有给出 next_db")
        if not decision.next_keyword:
            raise AnalyzeError("next_action 是 continue，但没有给出 next_keyword")


def _already_queried(state: AgentState, db: str | None, keyword: str | None) -> bool:
    """
    [WHY] 同一组 (库, 关键词) 再查一次必然拿到一样的资料 —— 检索是确定性的（BM25 + 向量，
          没有随机）。实测踩过：问「灭尽龙」（数据里没这只怪），模型第一轮就看出捞回来的
          是别的怪，却还是说 continue 去查同一个 (monster, 灭尽龙)，白烧一轮检索 + 一次 analyze。
    [FIX] 拿已经查过的组合比一比，比中就拦。这是纯字符串比较，不该交给模型"记得"
          （项目完善清单 §2.5）。关键词去首尾空格 —— 「灭尽龙」和「灭尽龙 」是同一个词。
    """
    wanted = (keyword or "").strip()
    for item in state["collected_info"]:
        if item["db"] == db and item["keyword"].strip() == wanted:
            return True
    return False


def _invoke_and_parse_analyze(user_content: str, retry_hint: str = "") -> AnalyzeDecision:
    """调一次模型并校验（AnalyzeDecision）；不合格抛 AnalyzeError，消息供上层回喂重试。"""
    content = user_content
    # 重试时把上次为什么不合格一并喂回去，不然模型会原样再错一遍
    if retry_hint:
        content += f"\n\n你上一次的输出无法使用，原因：{retry_hint}\n请按约定的字段重新输出。"

    # 系统提示词固定；用户内容 = 当前资料 +（重试时）上次的失败原因
    messages = [SystemMessage(content=ANALYZE_PROMPT), HumanMessage(content=content)]

    def call(llm: BaseChatModel) -> AnalyzeDecision:
        """
        [WHY] 必须显式写 method="json_mode"：DeepSeek 的思考模式模型（deepseek-flash）
              不支持 tool calling，报 "Thinking mode does not support this tool_choice"，
              而 with_structured_output 默认走的正是那条路 —— 主模型会直接 400。
        [FIX] json_mode 只绑 response_format={"type":"json_object"}，DeepSeek 支持。
              两个代价：① 没有 schema 约束、字段更容易漏 —— 靠上层的回喂重试兜；
                        ② DeepSeek 要求提示词里出现 "json" 字样，所以 ANALYZE_PROMPT
                          里保留了「输出 JSON 格式」那句，别删。
        """
        # 绑一次，之后 invoke 出来直接就是 AnalyzeDecision 对象，不用自己 json.loads
        bound = llm.with_structured_output(AnalyzeDecision, method="json_mode")
        try:
            return bound.invoke(messages)
        except (ValidationError, OutputParserException) as e:
            # 输出格式不合格是提示词/字段约定的问题，换模型也一样不合格 —— 该走回喂重试，
            # 所以包成 AnalyzeError 并告诉 invoke_with_fallback 别换模型
            raise AnalyzeError(f"模型输出不符合约定字段：{e}") from e

    # AnalyzeError 不换模型：输出格式的问题换谁都一样，该走上层的回喂重试
    decision, _used = invoke_with_fallback(call, no_fallback_on=(AnalyzeError,))
    # schema 表达不了的那条规则（continue 必须给去向），这里补一刀
    _check_decision(decision)
    return decision


def _decide(
    state: AgentState,
    source: str,
    action: str,
    db: str | None,
    keyword: str | None,
    reason: str,
    degraded: bool = False,
) -> dict:
    """把一次决策打包成节点返回值；同时写日志和 decision_trace（§2.4 §2.8）。"""
    # 一条决策的完整记录：日志和 decision_trace 用的是同一份
    entry = {
        "iteration": state["iteration_count"],
        "source": source,  # llm / code / degrade —— 这个决定是谁下的
        "action": action,
        "db": db,
        "keyword": keyword,
        "reason": reason,
    }
    logger.info("analyze 决策", extra=entry)

    # 节点返回值：告诉图停还是继续
    result = {
        "next_action": action,
        "degraded": degraded,
        "degrade_reason": reason if degraded else "",
        "decision_trace": [entry],
    }
    # 只有要继续才带下一站去哪儿；finish 时是 None
    if action == "continue":
        result["current_db"] = db
        result["query_keyword"] = keyword
    return result


def analyze_node(state: AgentState) -> dict:
    # 节点入口重绑编号：LangGraph 给每个节点的上下文各是一份，不会自动传过来
    bind_request_id(state.get("request_id", ""))
    # 给模型看三样：原始问题、已经查了几轮、手上有哪些资料
    collected_text = format_collected_info(state["collected_info"])
    user_content = f"""
原始问题：{state["user_question"]}
已检索轮次：{state["iteration_count"]}
已收集信息：
{collected_text}
"""

    # 重试状态：记着上一次为什么没成
    decision = None
    last_error = ""
    for attempt in range(ANALYZE_MAX_ATTEMPTS):
        try:
            decision = _invoke_and_parse_analyze(user_content, retry_hint=last_error)
            # 成了就跳出，不再试第二次
            break
        except AnalyzeError as e:
            # 格式类失败：记下原因，下一轮拼进提示词回喂
            last_error = str(e)
            logger.warning(
                "analyze 输出不合格", extra={"attempt": attempt + 1, "error": last_error}
            )
        except Exception as e:  # noqa: BLE001 —— 偶发故障也重来一次，仍失败才降级并留痕
            # 原来这里是 break（理由写着"重问一遍没用"）—— 但那是错的：
            # SDK 的 max_retries 只重试超时/5xx，400 这类它不管，我们不重试就只剩降级。
            # 踩过一次：DeepSeek 偶发 400（"response_format 现在不可用"），整轮直接降级，
            # 只查了 2 轮就出答案。
            last_error = f"调用失败：{type(e).__name__}: {e}"
            logger.warning("analyze 调用失败", extra={"attempt": attempt + 1, "error": last_error})

    # 两次都没成 —— 降级成 finish，但原因留在 trace 里，评测看得见
    if decision is None:
        return _decide(state, "degrade", "finish", None, None, last_error, degraded=True)

    # 代码兜底：这一组查过了就别再查 —— 同样的输入必然同样的结果，重复一轮纯属白跑。
    # 只拦"一模一样"的；换个词、换个库照走，那些是真能拿到新资料的。
    if decision.next_action == "continue" and _already_queried(
        state, decision.next_db, decision.next_keyword
    ):
        return _decide(
            state,
            "code",  # 这个决定是代码下的，不是模型下的 —— 事后能一眼分辨
            "finish",
            None,
            None,
            f"模型想接着查（{decision.next_db}, {decision.next_keyword}），"
            f"但这组已经查过了，重复查不会拿到新资料",
        )

    # 正常路径：把模型的决策原样交给 _decide 打包
    return _decide(
        state, "llm", decision.next_action, decision.next_db, decision.next_keyword, decision.reason
    )


# ── 条件边函数 ────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    """analyze 之后走哪条边（str）：轮次到顶或决定停就走 finish，否则 continue。"""
    # 轮次上限从配置读；读不到当 5，总之得有个顶，防死循环
    max_iterations = get_config().get("graph", {}).get("max_iterations", 5)
    # 先看轮次到没到顶
    if state["iteration_count"] >= max_iterations:
        logger.info("analyze 轮次到顶，强制 finish", extra={"iteration": state["iteration_count"]})
        return "finish"
    # 再看模型自己的意见
    if state["next_action"] == "finish":
        return "finish"
    return "continue"
