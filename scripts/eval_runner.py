"""
agent/scripts/eval_runner.py —— 跑金标 QA 集，统计通过率

用法（在仓库根目录 C:\\code\\MHWI_agent 下，用 -m 模块方式运行）：
    python -m agent.scripts.eval_runner

不能用 `python agent/scripts/eval_runner.py` 直接跑：那样 sys.path 只含
agent/scripts/，`from agent.graph import graph` 会报 ModuleNotFoundError。
"""

import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent.graph import graph


def main():
    qa_path = Path("tests/fixtures/golden_qa.json")
    golden = json.loads(qa_path.read_text(encoding="utf-8"))

    total = 0
    passed = 0
    for case in golden:
        q = case["question"]
        must = case.get("must_mention", [])
        must_not = case.get("must_not_contain", [])

        try:
            state = graph.invoke({
                "user_question": q,
                "messages": [HumanMessage(content=q)],
            })
            answer = state.get("final_answer", "")
        except Exception as e:
            print(f"[FAIL] {q} 运行异常: {e}")
            total += 1
            continue

        ok = True
        for m in must:
            if m not in answer:
                ok = False
                print(f"[FAIL] {q} 缺少关键词 {m}")
        for m in must_not:
            if m in answer:
                ok = False
                print(f"[FAIL] {q} 出现违禁词 {m}")

        total += 1
        if ok:
            passed += 1
        else:
            print(f"    回答片段: {answer[:120]}")

    rate = passed / total * 100 if total else 0
    print(f"\n金标 QA 通过率: {passed}/{total} = {rate:.1f}%")
    if rate < 80:
        sys.exit(1)


if __name__ == "__main__":
    main()
