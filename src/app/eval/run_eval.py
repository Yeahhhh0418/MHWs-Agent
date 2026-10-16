import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from app.agent.graph import graph
from app.core.config import get_config
from app.core.metrics import summary

EVAL_DIR = Path(__file__).parent
# 字段说明见同目录 README.md


def calc_recall(required_ids: list, retrieved_ids: list) -> tuple:
    """算召回率：该命中的编号里，实际命中了几个。返回（召回率, 没命中的编号）。"""
    # 逐条比对：该命中的编号，命中进 hit，没命中进 missing
    hit = []
    missing = []
    for one_id in required_ids:
        if one_id in retrieved_ids:
            hit.append(one_id)
        else:
            missing.append(one_id)

    # 空题目防除零，直接判 0 分
    if len(required_ids) == 0:
        recall = 0.0
    else:
        recall = len(hit) / len(required_ids)

    return recall, missing


def calc_reciprocal_rank(required_ids: list, rounds: list) -> tuple:
    """
    [WHY] 召回率只说「有没有捞到」，不说「捞到得多靠前」—— 排第 1 位和排第 5 位，
          召回率都是 100%，但检索质量差得远。
    [REF] MRR（Mean Reciprocal Rank，平均倒数排名）
    返回（倒数排名, 首次命中的第几轮, 第几位）；没命中返回 (0.0, None, None)。
    """
    # 各轮拼成一条长队：第 1 轮的先，第 2 轮的后 —— 早一轮捞到就是更好
    position = 0
    for round_index, one_round in enumerate(rounds):
        # 轮内本来就是按融合分数从高到低排的
        for one_id in one_round:
            position += 1
            if one_id in required_ids:
                return 1.0 / position, round_index + 1, position
    return 0.0, None, None


def calc_route_accuracy(required_ids: list, routed_dbs: list) -> tuple:
    """
    [WHY] 路由错了，后面检索再准也白搭 —— 该查怪物库却去了武器库，召回率必然是 0，
          但光看召回率分不清是路由错还是检索错。
    返回（该查的库是否都查了, 该查却没查的库, 查了但不需要的库）。
    """
    # 编号的前缀就是库名：monster|火龙|狩猎提示 -> monster
    required_dbs = []
    for one_id in required_ids:
        one_db = one_id.split("|")[0]
        if one_db not in required_dbs:
            required_dbs.append(one_db)

    # 该查却没查的 —— 这个才是真的错
    missing = []
    for one_db in required_dbs:
        if one_db not in routed_dbs:
            missing.append(one_db)

    # 查了但不该查的 —— 白跑的，记下来但不判错
    extra = []
    for one_db in routed_dbs:
        if one_db not in required_dbs:
            extra.append(one_db)

    return len(missing) == 0, missing, extra


def _git_commit() -> str:
    """
    当前 commit 的短 hash。**工作区有未提交改动时加个 `*`** ——
    否则这个 hash 指的是别人，不是你实际跑的代码。
    """
    try:
        # 拿当前 commit 的短 hash
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        # 再看工作区有没有没提交的改动
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 —— git 不在、不是仓库、命令超时……哪种都退回同一句话，抓全正好
        return "(不是 git 仓库)"
    # 工作区脏就打一个 *：提醒这个 hash 不完全代表实际跑的代码
    return f"{commit}*" if dirty else commit


def _config_snapshot(only_tags: list | None = None) -> dict:
    """
    [WHY] 结果文件不记当次配置 —— 过十天你就不知道那个数字是哪套参数跑出来的，
          所有"变好了"都证明不了（项目完善清单 §1.3）。
    [FIX] 把会影响结果的几样记下来：调过的参数、模型名、代码版本。
          提示词和数据也在影响结果，但它们分别由 git 和 data/ 管着。
    """
    # 读当前配置，把会影响结果的几项挑出来
    cfg = get_config()
    return {
        "ran_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "model": cfg["llm"]["primary"]["model"],
        "max_iterations": cfg["graph"]["max_iterations"],
        "retrieval": cfg["retrieval"],  # 各库的 k / 权重 / 阈值 / top_n
        # ⚠️ 只跑子集时必须记下来 —— 不然这份结果会被当成全量（题数对不上、指标也没法比）
        "only_tags": list(only_tags) if only_tags else [],
    }


def call_agent(question: str) -> dict:
    """
    返回值必须包含这几个字段（根据你的Agent实际返回调整）:
    - answer: 最终答案
    - retrieved_doc_ids: 检索到的文档ID列表（各轮拍平）
    - rounds: 每轮各自的编号列表，如 [["怪物|火龙|狩猎提示"], ["怪物|火龙|肉质"]]
    - round_counts: 每轮各返回几条，如 [10, 12, 4]
    - routed_dbs: 路由到的知识库名称列表
    - iterations: 迭代轮数
    """
    # 记录时间
    started = time.perf_counter()
    # 跑一遍 Agent，拿回完整状态
    st = graph.invoke({"messages": [HumanMessage(content=question)]})
    # 计算结束时间
    elapsed = time.perf_counter() - started
    # 取出每一轮的检索记录
    info = st.get("collected_info", [])

    # 把每轮的编号摊开：分轮的留一份，同时拍平成一个大列表
    rounds = []  # 每轮的编号列表
    doc_id_list = []
    for item in info:
        one_ids = item.get("doc_ids", [])
        rounds.append(one_ids)  # 整份留着，保住「这一轮有几条、都是谁」
        for one_id in one_ids:
            doc_id_list.append(one_id)

    # 收集这次用到了哪几个库，重复的不再记
    routed_dbs = []
    for item in info:
        if item["db"] not in routed_dbs:
            routed_dbs.append(item["db"])

    # 数一下每轮各返回几条
    round_counts = []
    for one_round in rounds:
        round_counts.append(len(one_round))

    usage = summary(st.get("request_id", ""))

    # 整理成评测脚本要的字段
    return {
        "answer": st.get("final_answer", ""),
        "retrieved_doc_ids": doc_id_list,
        "routed_dbs": routed_dbs,
        "iterations": st.get("iteration_count", 0),
        # analyze 每轮由谁定的、定了什么、为什么 + 有没有降级（项目完善清单 §2.4 §2.8）
        "decision_trace": st.get("decision_trace", []),
        "degraded": st.get("degraded", False),
        "rounds": rounds,
        "round_counts": round_counts,
        "elapsed_seconds": elapsed,  # 端到端耗时：进图到出答案
        "token_usage": usage,  # 这次问答的 token 用量，含 by_model 明细
    }


def _run_one(case: dict, snapshot: dict) -> dict:
    """跑一道题，返回一条结果记录（dict）。"""
    # 跑一道题
    agent_output = call_agent(case["question"])

    # 题库里要求命中的编号
    required_ids = case.get("required_doc_ids", [])
    # Agent 实际检出的编号
    retrieved_ids = agent_output.get("retrieved_doc_ids", [])
    # 比一比：召回率多少、漏了哪几个
    recall, missing = calc_recall(required_ids, retrieved_ids)

    # 分轮编号，算「捞到得多靠前」用
    rounds = agent_output.get("rounds", [])
    reciprocal_rank, first_hit_round, first_hit_position = calc_reciprocal_rank(
        required_ids, rounds
    )

    # 该查的库都查了没有
    routed_dbs = agent_output.get("routed_dbs", [])
    route_hit, route_missing, route_extra = calc_route_accuracy(required_ids, routed_dbs)

    # 拼成一条结果记录，落盘用
    return {
        **case,  # 保留原始题目信息
        "config": snapshot,  # ← 当次配置，见 _config_snapshot
        "agent_answer": agent_output.get("answer", ""),
        "retrieved_doc_ids": retrieved_ids,
        "rounds": rounds,  # MRR 要用：每条编号在第几轮的第几位
        "round_counts": agent_output.get("round_counts", []),  # 每轮返回几条，如 [10, 12, 4]
        "elapsed_seconds": agent_output.get("elapsed_seconds", 0.0),  # 端到端耗时（秒）
        "token_usage": agent_output.get("token_usage", {}),  # token 用量，含 by_model 明细
        "routed_dbs": routed_dbs,
        "iterations": agent_output.get("iterations", 0),
        "recall": recall,
        "missing_doc_ids": missing,
        "reciprocal_rank": reciprocal_rank,  # 该命中的那条排第几位的倒数
        "first_hit_round": first_hit_round,  # 首次命中落在第几轮
        "first_hit_position": first_hit_position,  # 拼成长队后的第几位
        "route_hit": route_hit,  # 该查的库都查了没有
        "route_missing": route_missing,  # 该查却没查的库
        "route_extra": route_extra,  # 查了但不需要的库
        "decision_trace": agent_output.get("decision_trace", []),
        "degraded": agent_output.get("degraded", False),
        # 人工判分预留字段
        "human_score": None,
        "fail_category": None,  # 可选: recall/premature_stop/routing/iteration/degrade/fact
    }


def run_eval(only_tags: list | None = None) -> None:
    """跑评测。only_tags 给了就只跑 tags 命中其中任意一个的题（调参/消融时省时间）。"""
    # 1. 加载题库
    # 一行一道题，读成 dict 列表，空行跳过
    cases = []
    with open(EVAL_DIR / "cases.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    # 只想跑某个场景的子集时（调参、消融），按 tags 筛 ——
    # 全量 31 道要 7 分钟，而一次消融往往只关心其中几道
    if only_tags:
        kept = []
        for one_case in cases:
            hit = False
            for one_tag in one_case.get("tags", []):
                if one_tag in only_tags:
                    hit = True
            if hit:
                kept.append(one_case)
        cases = kept
        print(f"只跑 tags 含 {only_tags} 的题：{len(cases)} 道")

    # 2. 先建好结果文件，然后**边跑边写**
    #    原来是攒在内存里、最后一次性写盘 —— 第 10 题崩了，前 9 条也白跑
    # 用时间戳当文件名，一次跑出一个新文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(EVAL_DIR / "results", exist_ok=True)
    output_path = EVAL_DIR / "results" / f"{timestamp}.jsonl"
    # 先记下这次的配置，随每条结果一起存
    snapshot = _config_snapshot(only_tags)

    results = []
    with open(output_path, "w", encoding="utf-8") as out:
        # 逐题跑
        for i, case in enumerate(cases):
            print(f"[{i + 1}/{len(cases)}] 正在测试: {case['question']}")
            try:
                # 正常跑完：拿到记录，顺手打印这题的指标
                record = _run_one(case, snapshot)
                # 成本压成一个短串（耗时秒 / token 千），免得这行太长
                one_usage = record.get("token_usage") or {}
                cost = (
                    f"{record.get('elapsed_seconds', 0.0):.0f}s/"
                    f"{one_usage.get('total_tokens', 0) / 1000:.1f}k"
                )
                print(
                    f"   召回率: {record['recall'] * 100:.0f}%   "
                    f"返回条数: {record.get('round_counts', [])}   "
                    f"倒数排名: {record.get('reciprocal_rank', 0.0):.2f}   "
                    f"成本 {cost}   "
                    f"没找到的编号: {record['missing_doc_ids']}"
                )
            except Exception as e:  # noqa: BLE001 —— 就是要抓全：任何一道题崩了都不能拖垮整轮评测
                # 一道题崩了不能拖垮整轮 —— 记下原因，接着跑下一道
                print(f"   ❌ 崩了: {type(e).__name__}: {e}")
                record = {**case, "config": snapshot, "error": f"{type(e).__name__}: {e}"}

            # 一题一行写盘；flush 保证立刻写进磁盘，后面再崩前面跑过的也保得住
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            results.append(record)

    # 3. 汇总
    # 把各题召回率加起来求平均；崩掉的题没有 recall 字段，按 0 分算
    total_recall = 0.0
    for r in results:
        total_recall += r.get("recall", 0.0)
    if len(results) > 0:
        avg_recall = total_recall / len(results)
    else:
        avg_recall = 0.0

    # MRR：各题倒数排名的平均。1.00 = 每题该命中的那条都排第一
    total_rr = 0.0
    for r in results:
        total_rr += r.get("reciprocal_rank", 0.0)
    if len(results) > 0:
        avg_mrr = total_rr / len(results)
    else:
        avg_mrr = 0.0

    # 路由准确率：该查的库都查了的题占几成
    route_hit_count = 0
    for r in results:
        if r.get("route_hit", False):
            route_hit_count += 1
    if len(results) > 0:
        route_accuracy = route_hit_count / len(results)
    else:
        route_accuracy = 0.0

    # 降级率：降级过的答案质量不能信，这个数越低越好
    degraded_count = 0
    for r in results:
        if r.get("degraded", False):
            degraded_count += 1

    # 首次命中落在第几轮：分布比平均值更能说明问题
    round_dist = {}
    for r in results:
        one_round = r.get("first_hit_round")
        if one_round is not None:
            round_dist[one_round] = round_dist.get(one_round, 0) + 1

    # 成本：整轮一共花了多少 token、多少时间
    total_tokens = 0
    total_reasoning = 0
    total_seconds = 0.0
    for r in results:
        one_usage = r.get("token_usage") or {}
        total_tokens += one_usage.get("total_tokens", 0)
        total_reasoning += one_usage.get("reasoning_tokens", 0)
        total_seconds += r.get("elapsed_seconds", 0.0)

    # 防除零：一条用量都没记上时，比例没意义
    if total_tokens > 0:
        reasoning_ratio = total_reasoning / total_tokens
    else:
        reasoning_ratio = 0.0
    if len(results) > 0:
        avg_seconds = total_seconds / len(results)
    else:
        avg_seconds = 0.0

    # 挑出崩掉的题，单独提醒一句
    crashed = []
    for r in results:
        if "error" in r:
            crashed.append(r)
    if crashed:
        print(f"⚠️ {len(crashed)} 道题崩了（按 0 分计入平均）")

    # 最后给一组总览
    print(f"平均召回率: {avg_recall * 100:.0f}%")
    print(f"平均 MRR: {avg_mrr:.2f}   （1.00 = 每题该命中的那条都排第一）")
    print(f"路由准确率: {route_accuracy * 100:.0f}%   （该查的库都查了的题占几成）")
    print(f"降级率: {degraded_count}/{len(results)}   （降级过的答案不能信）")
    print(f"首次命中的轮次分布: {round_dist}   # 如 {{1: 12, 2: 3}} = 12 题第一轮就捞到")
    # 成本两行：token 顺带报 reasoning 占比（思考模式的大头），耗时标清是各题相加
    print(f"总 token: {total_tokens}   （reasoning {total_reasoning}，占 {reasoning_ratio:.0%}）")
    print(f"耗时: 各题相加 {total_seconds / 60:.1f} 分钟   （平均每题 {avg_seconds:.0f} 秒）")
    print(f"✅ 评测完成，结果已保存至: {output_path}")


if __name__ == "__main__":
    # 用法：
    #   run_eval.py                  跑全部（31 道，约 7 分钟）
    #   run_eval.py 查族名,俗称        只跑 tags 命中其中任意一个的题
    tags = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    run_eval(tags)
