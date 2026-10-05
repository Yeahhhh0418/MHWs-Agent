"""
src/app/agent/nodes/common.py —— 多个节点共用的 LLM 调用与文本拼装。
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm


def invoke(system_prompt: str, user_content: str) -> str:
    """单次 LLM 调用，返回文本（str）。"""
    llm = get_llm()
    return llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
    ).content


def format_collected_info(collected_info: list) -> str:
    """把累积的检索结果拼成给模型看的参考资料文本（str）；空列表返回一句「没查到」。"""
    if not collected_info:
        return "未检索到任何相关信息。"
    formatted_text = "【已检索到的参考资料】：\n"
    for i, result in enumerate(collected_info, 1):
        formatted_text += f"{i}. 数据库: {result['db']}\n"
        formatted_text += f"   关键词: {result['keyword']}\n"
        formatted_text += f"   内容: {result['content']}\n\n"
    return formatted_text
