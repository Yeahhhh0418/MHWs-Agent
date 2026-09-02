import sys
import os

# 把根目录加入 Python 搜索路径，这样就能导入根目录下的 rag 文件
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from state import AgentState, RAGResult
from chatllm import llm
from prompts.prompt import ROUTE_PROMPT, ANALYZE_PROMPT, GENERATE_PROMPT
import json
from Equipment_Crafting_rag import EquipmentCraftingRAG
from ItemData_rag import ItemDataRAG
from MonsterData_rag import MonsterDataRAG


# 路由节点：根据用户问题，决定从哪个库开始查
def route_node(state: AgentState) -> dict:
    user_msg = state["messages"][-1].content

    response = llm.invoke([
        SystemMessage(content=ROUTE_PROMPT),
        HumanMessage(content=user_msg)
    ])

    result = json.loads(response.content)
    # result = {"first_db": "equipment", "first_keyword": "锁刃龙的头"}

    return {
        "user_question": user_msg,
        "current_db": result["first_db"],
        "query_keyword": result["first_keyword"],
        "iteration_count": 0,
        "collected_info": [],
    }

# ── RAG 实例懒加载缓存 ──
_rag_cache: dict[str, object] = {}

def _get_rag(db: str):
    """懒加载获取 RAG 实例，首次调用时创建并 setup，后续直接复用"""
    if db in _rag_cache:
        return _rag_cache[db]

    print(f"首次加载 {db} 数据库，正在初始化...")
    if db == "equipment":
        rag = EquipmentCraftingRAG()
    elif db == "item":
        rag = ItemDataRAG()
    elif db == "monster":
        rag = MonsterDataRAG()
    else:
        raise ValueError(f"未知数据库: {db}")

    rag.setup()
    _rag_cache[db] = rag
    return rag


# 检索节点：根据 current_db 和 query_keyword 去对应的库检索
def retrieve_node(state: AgentState) -> dict:
    db = state["current_db"]
    keyword = state["query_keyword"]
    iteration = state.get("iteration_count", 0)

    rag = _get_rag(db)
    print(f"正在调用{db}数据库")
    content = rag.query(keyword)

    new_result = RAGResult(db=db, keyword=keyword, content=content)

    return {
        "collected_info": [new_result],
        "iteration_count": iteration + 1,
    }

def format_collected_info(collected_info: list) -> str:
    """
    将 LangGraph 中累积的检索结果（字典列表）转换为大模型易读的纯文本
    """
    # 1. 如果列表为空，说明没查到东西
    if not collected_info:
        return "未检索到任何相关信息。"
    
    # 2. 遍历列表，把每一个字典拼接成字符串
    formatted_text = "【已检索到的参考资料】：\n"
    for i, result in enumerate(collected_info, 1):
        formatted_text += f"{i}. 数据库: {result['db']}\n"
        formatted_text += f"   关键词: {result['keyword']}\n"
        formatted_text += f"   内容: {result['content']}\n\n"
        
    return formatted_text
    
# 分析节点：判断信息够不够，决定要不要继续查
def analyze_node(state) -> dict:
    collected_text = format_collected_info(state["collected_info"])
    response = llm.invoke([
        SystemMessage(content=ANALYZE_PROMPT),
        HumanMessage(content=f"""
原始问题：{state["user_question"]}
已检索轮次：{state["iteration_count"]}
已收集信息：
{collected_text}

请判断是否需要继续检索，输出 JSON。
""")
    ])
    result = json.loads(response.content)
    # result = {
    #     "next_action": "continue" | "finish",
    #     "reason": "装备合成需要素材A，但素材A的获取方式未知，需要查item库",
    #     "next_db": "item",         # 仅 next_action="continue" 时
    #     "next_keyword": "素材A"     # 仅 next_action="continue" 时
    # }
    if result["next_action"] == "continue":
        return {
            "next_action": "continue",
            "current_db": result["next_db"],
            "query_keyword": result["next_keyword"],
        }
    else:
        return {"next_action": "finish"}


    
# 生成回答节点：把所有收集到的信息交给大模型生成最终回答
def generate_node(state: AgentState) -> dict:
    collected_text = format_collected_info(state["collected_info"])

    response = llm.invoke([
        SystemMessage(content=GENERATE_PROMPT),
        HumanMessage(content=f"""
玩家问题：{state["user_question"]}

以下是从数据库检索到的相关信息：
{collected_text}

请根据以上信息，用自然、友好的语气回答玩家的问题。
""")
    ])

    return {
        "final_answer": response.content,
        "messages": [AIMessage(content=response.content)],
    }

# 条件边函数
def should_continue(state: AgentState) -> str:
    MAX_ITERATIONS = 5

    if state["iteration_count"] >= MAX_ITERATIONS:
        return "finish"
    if state["next_action"] == "finish":
        return "finish"
    return "continue"


if __name__ == "__main__":
    pass