from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
import operator

class RAGResult(TypedDict):
    """单次 RAG 检索的结果"""
    db: str              
    keyword: str         # 本次检索的关键词
    content: str         # 检索到的内容（RAG 返回的文本）
    doc_ids: list[str]   # 这一轮命中的文档编号

class AgentState(TypedDict):
    # ── 对话 ──
    messages: Annotated[list, add_messages]
    user_question: str                    # 原始用户问题（第一次就存下来）

    # ── 当前检索轮次 ──
    current_db: str                       # 当前要查的库
    query_keyword: str                    # 当前检索关键词

    # ── 累积的检索轨迹（链式证据） ──
    collected_info: Annotated[list[RAGResult],operator.add]       # 每一轮检索的结果都追加到这里

    # ── 分析节点的决策输出 ──
    next_action: str                      # "continue" | "finish"
    # 当 next_action="continue" 时，以下字段同时被设置：
    #   current_db: 下一个要查的库
    #   query_keyword: 下一个要查的关键词

    # ── 安全阀 ──
    iteration_count: int                  # 已执行的检索次数，防止无限循环

    # ── 最终输出 ──
    final_answer: str                     # generate_node 写入


