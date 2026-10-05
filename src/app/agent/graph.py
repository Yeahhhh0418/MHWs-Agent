from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import analyze_node, generate_node, retrieve_node, route_node, should_continue
from app.agent.state import AgentState

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
    },
)

builder.add_edge("generate", END)

graph = builder.compile()

if __name__ == "__main__":
    while(1):
        a=input("请输入你的回答:  （exit退出）",) or "锁刃龙太刀和锁刃龙套怎么做？素材怎么获得？怎么打？"
        if a== "exit":
            break
        state = graph.invoke({"messages": [HumanMessage(content=a)]})
        print(state["final_answer"])