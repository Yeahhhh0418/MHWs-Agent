"""
命令行入口，对应 pyproject 里的 mhws-agent 命令。
用 `python -m app` 走的是同一条路（见 __main__.py）。
"""

import sys
import time

from langchain_core.messages import HumanMessage

from app.agent.graph import graph
from app.core.metrics import format_cost_line, summary


def main() -> None:
    """把命令行里的一句话丢给 Agent，打印最终答案；不带参数时只打印用法。"""
    if len(sys.argv) < 2:
        print('用法: mhws-agent "你的问题"')
        print('例子: mhws-agent "火龙这个怪要怎么打？"')
        return

    question = " ".join(sys.argv[1:])

    try:
        # 掐表：从进图到出答案
        started = time.perf_counter()
        state = graph.invoke({"messages": [HumanMessage(content=question)]})
        elapsed = time.perf_counter() - started
    except Exception as e:  # noqa: BLE001 —— 顶层兜底：不让异常裸奔给用户，必须抓全
        # 失败要吼出来，不能装作答完了（比如 .env 里缺 API Key）
        print(f"运行失败：{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    print(state.get("final_answer", "（没有拿到答案）"))
    # 降级要吼一声：答案是建立在残缺资料上的，得让用户知道（检索失败 / 生成失败都会走到这儿）
    if state.get("degraded"):
        print(f"⚠️ 这次有步骤降级了：{state.get('degrade_reason') or '原因未记录'}", file=sys.stderr)
    # 成本走 stderr：stdout 只留答案，用户可能把它重定向去用
    usage = summary(state.get("request_id", ""))
    print(f"[{format_cost_line(usage, elapsed)}]", file=sys.stderr)
    print(f"[本次会话编号：{state.get('request_id', '-')}]", file=sys.stderr)
