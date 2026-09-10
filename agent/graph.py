from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from agent.nodes import route_node, retrieve_node, analyze_node, generate_node, should_continue
from agent.state import AgentState
from agent.tracker import AgentTracker, set_tracker

# 构建图
builder = StateGraph(AgentState)

builder.add_node("route", route_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("analyze", analyze_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "route")
builder.add_edge("route", "retrieve")
builder.add_edge("retrieve", "analyze")

builder.add_conditional_edges(
    "analyze",
    should_continue,
    {
        "continue": "retrieve",
        "finish": "generate",
    }
)

builder.add_edge("generate", END)

graph = builder.compile()


def run_with_tracking(user_question: str) -> dict:
    """带追踪地跑一次完整查询。即使崩溃也会把 trace 写进日志。"""
    tracker = AgentTracker()
    set_tracker(tracker)
    try:
        return graph.invoke({
            "user_question": user_question,
            "messages": [HumanMessage(content=user_question)],
        })
    finally:
        tracker.flush_to_log()  # 无论成功失败，都记录
        print(tracker.summarize())  # 人类可读摘要，一眼看出几轮/最慢节点


if __name__ == "__main__":
    user_question = "巨戟龙的头要怎么做，需要什么素材，怎么获得需要的素材？"
    final_state = run_with_tracking(user_question)
    print(final_state["final_answer"])
