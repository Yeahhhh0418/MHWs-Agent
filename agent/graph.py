from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from nodes import route_node, retrieve_node, analyze_node, generate_node, should_continue
from state import AgentState
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
    should_continue,        # 条件函数
    {
        "continue": "retrieve",  # 继续循环
        "finish": "generate",    # 结束，生成回答
    }
)

builder.add_edge("generate", END)

graph = builder.compile()

if __name__ == "__main__":
    # 1. 准备初始输入（字典格式）
    user_question = "巨戟龙的头要怎么做，需要什么素材，怎么获得需要的素材？"
    initial_input = {
        "user_question": user_question,
        "messages": [HumanMessage(content=user_question)]  # route_node 会读 messages[-1]
    }

    # 2. 运行图
    final_state = graph.invoke(initial_input)

    # 3. 打印最终生成的答案
    print(final_state["final_answer"])