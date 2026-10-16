import operator
from typing import Annotated, Literal, TypeAlias, TypedDict

from langgraph.graph.message import add_messages

# 四个库的名字。route 和 analyze 都拿它约束模型的输出（非法库名当场打回），
# 所以放在这里一份 —— 两处各写一遍的话，加库时漏掉一处就是静默漂开。
DB: TypeAlias = Literal["equipment", "weapon", "material", "monster"]


class RAGResult(TypedDict):
    """单次 RAG 检索的结果"""

    db: str
    keyword: str  # 本次检索的关键词
    content: str  # 检索到的内容（RAG 返回的文本）
    doc_ids: list[str]  # 这一轮命中的文档编号


class AgentState(TypedDict):
    # ── 对话 ──
    messages: Annotated[list, add_messages]
    user_question: str  # 原始用户问题（第一次就存下来）

    # ── 当前检索轮次 ──
    current_db: str  # 当前要查的库
    query_keyword: str  # 当前检索关键词

    # ── 累积的检索轨迹（链式证据） ──
    collected_info: Annotated[list[RAGResult], operator.add]  # 每一轮检索的结果都追加到这里

    # ── 分析节点的决策输出 ──
    next_action: str  # "continue" | "finish"
    # 当 next_action="continue" 时，以下字段同时被设置：
    #   current_db: 下一个要查的库
    #   query_keyword: 下一个要查的关键词

    # ── 安全阀 ──
    iteration_count: int  # 已执行的检索次数，防止无限循环
    # 这次问答的起点（time.monotonic()，route 时记）。analyze 拿它算已经跑了多久 ——
    # iteration_count 管"转几轮"，这个管"总共花多久"；每轮都慢的时候只有它能兜住。
    started_at: float

    # ── 最终输出 ──
    final_answer: str  # generate_node 写入

    # ── 降级留痕 ──
    # ⚠️ 必须用 or_ 合并：普通字段是"后写的冲掉先写的"，而每个节点都会写这个字段，
    #    所以 route 降级了、后面 retrieve 写个 False，降级就被吞掉了（实测过）。
    #    用 or_ 之后语义是"这次问答有没有任何一步降级过" —— 正是降级率要的。
    degraded: Annotated[bool, operator.or_]
    # ⚠️ 配套的约定：这个字段**只在真降级时才写**。它没有 reducer，谁写谁覆盖，
    #    所以要是不降级也写个空串，会把前一个节点的原因冲掉。
    degrade_reason: str

    # ── 决策轨迹：每一轮 analyze 由谁定的、定了什么、为什么 ──
    # 每条形如 {"iteration": 1, "source": "llm", "action": "continue",
    #          "db": "monster", "keyword": "火龙", "reason": "..."}
    decision_trace: Annotated[list[dict], operator.add]

    # ── 本次问答的编号：串日志用，出问题时按它捞整条链路 ──
    request_id: str
