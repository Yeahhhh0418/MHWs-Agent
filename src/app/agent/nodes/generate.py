"""
src/app/agent/nodes/generate.py —— 生成节点：把收集到的资料组织成人话。
"""

from langchain_core.messages import AIMessage

from app.agent.nodes.common import format_collected_info, invoke
from app.agent.state import AgentState
from app.logger import get_logger
from app.prompts import GENERATE_PROMPT, LIBRARY_GUIDES

logger = get_logger("nodes.generate")


def _library_guides(collected_info: list) -> str:
    """把这次检索命中过的库对应的规则拼起来（str）；一个都没命中就返回空串。"""
    used_dbs = []
    for item in collected_info:
        if item["db"] not in used_dbs:
            used_dbs.append(item["db"])

    parts = []
    for db in used_dbs:
        guide = LIBRARY_GUIDES.get(db)
        if guide:
            parts.append(guide)
    return "\n\n".join(parts)


def generate_node(state: AgentState) -> dict:
    collected_text = format_collected_info(state["collected_info"])
    guides = _library_guides(state["collected_info"])
    system_prompt = GENERATE_PROMPT
    if guides:
        system_prompt = f"{GENERATE_PROMPT}\n\n{guides}"
    try:
        answer = invoke(
            system_prompt,
            f"""
玩家问题：{state["user_question"]}
以下是从数据库检索到的相关信息：{collected_text}
""",
        )
    except Exception as e:  # noqa: BLE001 —— 生成失败就退回检索原文，至少用户还能看到资料
        logger.error("生成失败，返回原始检索内容", extra={"error": str(e)})
        answer = collected_text

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }
