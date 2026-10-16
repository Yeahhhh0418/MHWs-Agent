"""
src/app/eval/judge_facts.py —— 答案召回率：拿 key_facts 去答案里逐条找。

跟 run_eval.py 分开：这个读**已有的**结果文件，不重跑 Agent（省掉最贵的那部分）。
"""

import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.core.llm import invoke_with_fallback

EVAL_DIR = Path(__file__).parent


class FactVerdict(BaseModel):
    """一条事实点的判定结果。"""

    fact: str = Field(description="事实点原文，照抄给的那条")
    appeared: bool = Field(description="这个意思在回答里出现了吗")
    evidence: str = Field(description="回答里支撑这个判断的原句；没出现就留空字符串")


class JudgeResult(BaseModel):
    """一次判分的全部结果。"""

    verdicts: list[FactVerdict] = Field(description="每条事实点一条判定")


JUDGE_PROMPT = """你是一个评测助手。判断下面几个「事实点」有没有出现在这段回答里。

【判断标准】看的是**意思到没到**，不是字面一样：

- 回答用别的说法表达了同样的意思 → 算出现
  （例：事实点「用闪光使其目眩」，回答写「带闪光弹可以把它打下来」→ 算出现）
- 回答只说了事实点的一半 → 算没出现
- 回答说了相反的意思 → 算没出现

## 输出 JSON 格式

只输出一个 JSON 对象，不要有任何多余文字。顶层字段就叫 `verdicts`，
**一个事实点一条、条数和顺序都跟给的 facts 对齐**（少一条这题就没法算分了）：

```json
{"verdicts": [{"fact": "事实点原文，照抄", "appeared": true, "evidence": "从回答里摘出的原句"}]}
```

- fact：事实点原文，照抄
- appeared：true / false
- evidence：从回答里摘出支撑这个判断的原句；appeared 是 false 就填空字符串 ""
"""


def judge_facts(question: str, answer: str, key_facts: list) -> tuple[JudgeResult, str]:
    """
    [WHY] 召回率只回答「资料捞到了没有」，回答不了「答案写对了没有」——
          资料在、模型却没写进去的情况，现在完全看不见。
    [FIX] 拿题库里的 key_facts（出题时人从资料里摘的必答点）去答案里逐条找，
          算一个「答案召回率」。
    返回（判分结果, 判官模型名）—— 判官是谁得看得见：裁判换了，分数就没法跟上一批比。
    """
    # 一次调用判完整题：事实点一般 2~5 条，分次调太碎
    lines = []
    for one_fact in key_facts:
        lines.append(f"- {one_fact}")
    fact_text = "\n".join(lines)

    # 带上问题：事实点脱离语境可能看不懂（同一个词在两题里指的不是一回事）
    user_content = f"问题：{question}\n\n回答：{answer}\n\n要判的事实点：\n{fact_text}\n"
    messages = [SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=user_content)]

    def call(llm):
        # json_mode 跟 analyze 那边一样 —— DeepSeek 的思考模式不支持 tool calling
        bound = llm.with_structured_output(JudgeResult, method="json_mode")
        return bound.invoke(messages)

    result, judge_model = invoke_with_fallback(call)
    return result, judge_model


def run_judge(result_file: str = "") -> None:
    """
    读一份已有的结果文件，逐题判分，打印 + 另存一份带判分的（不覆盖原文件）。
    [WHY] 判分要真调模型，31 题是一整轮的钱和时间，攒到最后一次性写盘，
          中间断了（超时、上下文爆、某题卡住）就一条都不剩，全部白跑。
    [FIX] 跟 run_eval.py 一样边判边写，每题落一行。
    """
    # 没指定文件名就挑最新那份
    if not result_file:
        files = []
        # 带判分的 .judged.jsonl 跳过 —— 那是这个脚本自己的产物，不是评测结果
        for one in sorted((EVAL_DIR / "results").glob("*.jsonl")):
            if not one.name.endswith(".judged.jsonl"):
                files.append(one)
        if not files:
            print("results/ 下没有结果文件")
            return
        result_file = str(files[-1])
    src_path = Path(result_file)

    # 一行一题读进来
    records = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # 另存：带判分的那份。不覆盖原文件 —— 原文件是"跑评测的记录"，判分是"另一层分析"
    out_path = src_path.parent / f"{src_path.stem}.judged.jsonl"

    print(f"读入 {len(records)} 题：{src_path.name}")
    print()

    total_facts = 0
    total_appeared = 0
    # 这次判分用过哪些判官、各判了几题 —— 一轮里混了两个就等于两把尺子
    judge_counts = {}
    with open(out_path, "w", encoding="utf-8") as out:
        for i, record in enumerate(records):
            question = record.get("question", "")
            answer = record.get("agent_answer", "")
            key_facts = record.get("key_facts", [])

            # 默认原样落盘，判成功才往上叠判分字段 ——
            # 跳过的题也写一行，下游才分得清"这题没判成"和"这题压根不在题库里"
            new_record = dict(record)

            print(f"[{i + 1}/{len(records)}] {question[:28]}")

            # 下面的 continue 是这题"没判成"的几种出路，finally 兜住它们，
            # 保证不管走哪条都落一行盘
            try:
                # 崩掉的题没有答案，跳过 —— 它不该算进覆盖率
                if "error" in record or not key_facts:
                    reason = "这题崩了" if "error" in record else "没写 key_facts"
                    print(f"        跳过（{reason}）")
                    print()
                    continue

                try:
                    result, judge_model = judge_facts(question, answer, key_facts)
                except Exception as e:  # noqa: BLE001 —— 一题判分失败不该拖垮整轮
                    print(f"        ❌ 判分失败: {type(e).__name__}: {e}")
                    print()
                    continue

                # ⚠️ json_mode 没有 schema 约束，模型可能少给条数 —— 对不上就当这题判分不可信，不计入
                if len(result.verdicts) != len(key_facts):
                    print(
                        f"        ⚠️ 条数对不上（给了 {len(key_facts)} 条，回来 {len(result.verdicts)} 条），这题不计入"
                    )
                    print()
                    continue

                # 数这一题命中了几个事实点，顺便把没命中的打出来
                appeared_count = 0
                for one in result.verdicts:
                    if one.appeared:
                        appeared_count += 1
                total_facts += len(key_facts)
                total_appeared += appeared_count
                one_recall = appeared_count / len(key_facts)

                # 逐题记下判官：混判在末尾汇总，这里能看出是从哪一题换的人
                judge_counts[judge_model] = judge_counts.get(judge_model, 0) + 1

                print(
                    f"        事实点 {len(key_facts)} 条 -> {appeared_count} 条出现        答案召回 {one_recall * 100:.0f}%        [判官 {judge_model}]"
                )
                for one in result.verdicts:
                    if not one.appeared:
                        print(f"           ✗ {one.fact}")
                print()

                # 判分结果并进原记录（原文件不动）
                verdict_list = []
                for one in result.verdicts:
                    verdict_list.append(one.model_dump())
                new_record["fact_verdicts"] = verdict_list
                new_record["answer_recall"] = one_recall
                # 判官落盘：终端关了之后，还能核对两份分数是不是同一个判官给的
                new_record["judge_model"] = judge_model
            finally:
                # 一题一行写盘；flush 保证立刻进磁盘，后面再断，前面判过的也保得住
                out.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                out.flush()

    print("-" * 56)
    if total_facts > 0:
        print(
            f"答案召回率: {total_appeared / total_facts * 100:.0f}%   "
            f"（{total_appeared} / {total_facts} 条事实点）"
        )
    else:
        print("答案召回率: 无（没有可判的事实点）")

    # 判官是谁必须报出来 —— 一批分数只有同一个判官判出来才谈得上互相比较
    if judge_counts:
        parts = []
        for one_model, count in judge_counts.items():
            parts.append(f"{one_model}×{count}")
        judge_line = "判官模型: " + "、".join(parts)
        # 混判是静默发生的（主模型偶尔挂一次就换备用上场），所以这里要吼
        if len(judge_counts) > 1:
            judge_line += "   ⚠️ 混了多个判官，这批分数不能当成同一把尺子"
        print(judge_line)

    print(f"✅ 判分结果已另存至: {out_path}")


if __name__ == "__main__":
    # 用法：
    #   judge_facts.py                    判最新那份结果
    #   judge_facts.py results/xxx.jsonl   判指定的那份
    file_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    run_judge(file_arg)
