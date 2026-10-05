import json
import os
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from app.agent.graph import graph

EVAL_DIR = Path(__file__).parent
# 字段说明见同目录 README.md


def calc_recall(required_ids: list, retrieved_ids: list) -> tuple:
    """算召回率：该命中的编号里，实际命中了几个。返回（召回率, 没命中的编号）。"""
    hit = []
    missing = []
    for one_id in required_ids:
        if one_id in retrieved_ids:
            hit.append(one_id)
        else:
            missing.append(one_id)

    if len(required_ids) == 0:
        recall = 0.0
    else:
        recall = len(hit) / len(required_ids)

    return recall, missing


# TODO: 替换成你真实的 Agent 调用函数
def call_agent(question: str) -> dict:
    """
    返回值必须包含这四个字段（根据你的Agent实际返回调整）:
    - answer: 最终答案
    - retrieved_doc_ids: 检索到的文档ID列表
    - routed_dbs: 路由到的知识库名称列表
    - iterations: 迭代轮数
    """
    # 这里放你的 Agent 代码
    st = graph.invoke({"messages": [HumanMessage(content=question)]})
    info = st.get("collected_info", [])

    doc_id_list = []
    for item in info:
        one_ids = item.get("doc_ids", [])
        for one_id in one_ids:
            doc_id_list.append(one_id)

    routed_dbs = []
    for item in info:
        if item["db"] not in routed_dbs:
            routed_dbs.append(item["db"])

    return {
        "answer": st.get("final_answer", ""),
        "retrieved_doc_ids": doc_id_list,
        "routed_dbs": routed_dbs,
        "iterations": st.get("iteration_count", 0),
        # analyze 每轮由谁定的、定了什么、为什么 + 有没有降级（项目完善清单 §2.4 §2.8）
        "decision_trace": st.get("decision_trace", []),
        "degraded": st.get("degraded", False),
    }


def run_eval() -> None:
    # 1. 加载题库
    cases = []
    with open(EVAL_DIR / "cases.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    # 2. 逐条跑，记录过程
    results = []
    for i, case in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] 正在测试: {case['question']}")

        agent_output = call_agent(case["question"])

        required_ids = case.get("required_doc_ids", [])
        retrieved_ids = agent_output.get("retrieved_doc_ids", [])
        recall, missing = calc_recall(required_ids, retrieved_ids)
        print(f"   召回率: {recall * 100:.0f}%   没找到的编号: {missing}")

        record = {
            **case,  # 保留原始题目信息
            "agent_answer": agent_output.get("answer", ""),
            "retrieved_doc_ids": retrieved_ids,
            "routed_dbs": agent_output.get("routed_dbs", []),
            "iterations": agent_output.get("iterations", 0),
            "recall": recall,
            "missing_doc_ids": missing,
            "decision_trace": agent_output.get("decision_trace", []),
            "degraded": agent_output.get("degraded", False),
            # 人工判分预留字段
            "human_score": None,
            "fail_category": None,  # 可选: recall/premature_stop/routing/iteration/degrade/fact
        }
        results.append(record)

    # 3. 按时间戳存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(EVAL_DIR / "results", exist_ok=True)
    output_path = EVAL_DIR / "results" / f"{timestamp}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_recall = 0.0
    for r in results:
        total_recall += r["recall"]
    if len(results) > 0:
        avg_recall = total_recall / len(results)
    else:
        avg_recall = 0.0

    print(f"平均召回率: {avg_recall * 100:.0f}%")
    print(f"✅ 评测完成，结果已保存至: {output_path}")


if __name__ == "__main__":
    run_eval()
