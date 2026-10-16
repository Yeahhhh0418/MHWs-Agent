"""
src/app/agent/nodes/generate.py —— 生成节点：把收集到的资料组织成人话。
"""

from langchain_core.messages import AIMessage

from app.agent.nodes.common import format_collected_info, invoke
from app.agent.state import AgentState
from app.core.logger import bind_request_id, get_logger
from app.prompts import GENERATE_PROMPT, LIBRARY_GUIDES

logger = get_logger("nodes.generate")


def _library_guides(collected_info: list) -> str:
    """把这次检索命中过的库对应的规则拼起来（str）；一个都没命中就返回空串。"""
    # 先看这次碰过哪几个库，去重
    used_dbs = []
    for item in collected_info:
        if item["db"] not in used_dbs:
            used_dbs.append(item["db"])

    # 只拼配了规则的库 —— 不是每个库都有 guide
    parts = []
    for db in used_dbs:
        guide = LIBRARY_GUIDES.get(db)
        if guide:
            parts.append(guide)
    return "\n\n".join(parts)


def generate_node(state: AgentState) -> dict:
    # 节点入口重绑编号：LangGraph 给每个节点的上下文各是一份，不会自动传过来
    bind_request_id(state.get("request_id", ""))
    # 两样东西：检索到的资料正文，加上这次命中库的专门规则
    collected_text = format_collected_info(state["collected_info"])
    guides = _library_guides(state["collected_info"])
    # 有专门规则才接上去，没有就用通用提示词
    system_prompt = GENERATE_PROMPT
    if guides:
        system_prompt = f"{GENERATE_PROMPT}\n\n{guides}"
    # 默认没降级，下面出错时改写
    degraded = False
    degrade_reason = ""
    try:
        answer = invoke(
            system_prompt,
            f"""
玩家问题：{state["user_question"]}
以下是从数据库检索到的相关信息：{collected_text}
""",
        )
        logger.info("生成完成")
    except Exception as e:  # noqa: BLE001 —— 生成失败就退回检索原文，至少用户还能看到资料
        logger.error("生成失败，返回原始检索内容", extra={"error": str(e)})
        # 没润色，但资料是真的 —— 总比给用户一句"出错了"强。
        # ⚠️ 必须置 degraded：不置的话界面一切正常，用户拿到一屏资料原文还以为是答案，
        #    而真正的原因（比如密钥被拒）只躺在日志里（实测踩过：看着像"模型没接上"）
        answer = collected_text
        degraded = True
        degrade_reason = f"生成失败（{type(e).__name__}），下面是没加工过的原始检索资料"

    # final_answer 给评测和 CLI 读；messages 是图往下传的
    result = {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "degraded": degraded,
    }
    # 降级原因照样写：没降级时是空串，会被 merge_degrade_reason 忽略
    result["degrade_reason"] = degrade_reason
    return result
