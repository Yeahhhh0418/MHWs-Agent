"""
src/app/agent/nodes/common.py —— 多个节点共用的 LLM 调用与文本拼装。
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import invoke_with_fallback


def invoke(system_prompt: str, user_content: str) -> str:
    """单次 LLM 调用，返回文本（str）；主模型挂了自动换备用，切换会记日志。"""
    # 系统提示词 + 用户内容，所有节点都是这个调用形状
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    # 用了哪个模型这里不关心；要关心（比如想记进 trace）就自己调 invoke_with_fallback
    text, _used = invoke_with_fallback(lambda llm: llm.invoke(messages).content)
    return text


def format_collected_info(collected_info: list) -> str:
    """把累积的检索结果拼成给模型看的参考资料文本（str）；空列表返回一句「没查到」。"""
    # 一条都没查到就明说，别给模型留一段空白
    if not collected_info:
        return "未检索到任何相关信息。"
    # 每条写清：来自哪个库、用什么词查的、正文是什么
    formatted_text = "【已检索到的参考资料】：\n"
    # 编号让模型能指代"第几条"，跟日志里的轮次也对得上
    for i, result in enumerate(collected_info, 1):
        formatted_text += f"{i}. 数据库: {result['db']}\n"
        formatted_text += f"   关键词: {result['keyword']}\n"
        formatted_text += f"   内容: {result['content']}\n\n"
    return formatted_text
