"""
命令行入口，对应 pyproject 里的 mhwi-agent 命令。
用 `python -m mhwi_agent` 走的是同一条路（见 __main__.py）。
"""

import sys

from langchain_core.messages import HumanMessage

from agent.graph import graph


def main() -> None:
    """把命令行里的一句话丢给 Agent，打印最终答案；不带参数时只打印用法。"""
    if len(sys.argv) < 2:
        print('用法: mhwi-agent "你的问题"')
        print('例子: mhwi-agent "火龙这个怪要怎么打？"')
        return

    question = " ".join(sys.argv[1:])

    try:
        state = graph.invoke({"messages": [HumanMessage(content=question)]})
    except Exception as e:
        # 失败要吼出来，不能装作答完了（比如 .env 里缺 API Key）
        print(f"运行失败：{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    print(state.get("final_answer", "（没有拿到答案）"))
