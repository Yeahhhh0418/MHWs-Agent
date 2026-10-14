"""一次问答的耗时与 token 用量。"""

import copy
import time

from langchain_core.callbacks import BaseCallbackHandler

from app.logger import current_request_id

# request_id -> 这次问答的账。
# ⚠️ 必须是模块级 dict，不能用 contextvar —— LangGraph 给每个节点发独立的上下文副本，
#    contextvar 里攒的东西跨不了节点（见 logger.py 的 bind_request_id）
_ledger: dict = {}


def _empty_stats() -> dict:
    """一份全 0 的用量账；总账和 by_model 里的小账共用它，免得两边字段漂开。"""
    return {
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,  # 单列：思考模式的成本大头
        "total_tokens": 0,
        "llm_seconds": 0.0,  # 只算模型调用花的时间
    }


def _bucket(request_id: str) -> dict:
    """取（没有就新建）某个编号对应的账本。"""
    if request_id not in _ledger:
        stats = _empty_stats()
        # 模型名 -> 一份小账，用来区分主模型和备用模型
        stats["by_model"] = {}
        _ledger[request_id] = stats
    return _ledger[request_id]


def add_llm_call(model: str, usage: dict, seconds: float) -> None:
    """记一次模型调用；usage 为空（有些路拿不到）时只记调用次数和耗时。"""
    # 账挂在当前问答编号上 —— 编号由各节点入口的 bind_request_id 设好
    bucket = _bucket(current_request_id())
    bucket["llm_calls"] += 1
    bucket["llm_seconds"] += seconds

    # 按模型再记一份：跑慢了能立刻看出是备用模型拖的
    by_model = bucket["by_model"]
    if model not in by_model:
        by_model[model] = _empty_stats()
    one_model = by_model[model]
    # 次数和耗时先记上 —— 拿不到用量，它也是一次真实调用
    one_model["llm_calls"] += 1
    one_model["llm_seconds"] += seconds

    # 拿不到用量就到此为止，别把空值混进去
    if not usage:
        return

    # reasoning 嵌在 output_token_details 里，而且那层可能是 None
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    reasoning_tokens = (usage.get("output_token_details") or {}).get("reasoning", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    bucket["input_tokens"] += input_tokens
    bucket["output_tokens"] += output_tokens
    bucket["reasoning_tokens"] += reasoning_tokens
    bucket["total_tokens"] += total_tokens

    one_model["input_tokens"] += input_tokens
    one_model["output_tokens"] += output_tokens
    one_model["reasoning_tokens"] += reasoning_tokens
    one_model["total_tokens"] += total_tokens


def summary(request_id: str) -> dict:
    """取这次问答的总账（dict）；没有记录就返回一份全 0 的。"""
    if request_id not in _ledger:
        stats = _empty_stats()
        stats["by_model"] = {}
        return stats
    # 给的是副本：直接给引用的话，外面改一下就把账本本身改了
    return copy.deepcopy(_ledger[request_id])


def format_cost_line(usage: dict, elapsed_seconds: float) -> str:
    """把一次问答的成本压成一行给人看的文字（str）；CLI 和调试入口共用。"""
    # 用量一条都没记上时只报耗时，别甩一串 0 出来
    if not usage or usage.get("total_tokens", 0) == 0:
        return f"耗时 {elapsed_seconds:.0f} 秒（这次没记到 token 用量）"

    parts = [
        f"耗时 {elapsed_seconds:.0f} 秒",
        # 三段拼一句：括号写出来，省得看起来像三个独立的列表项
        (
            f"token {usage['total_tokens']}"
            f"（输入 {usage['input_tokens']} / 输出 {usage['output_tokens']}，"
            f"其中 reasoning {usage['reasoning_tokens']}）"
        ),
        f"{usage['llm_calls']} 次模型调用",
    ]
    # 把用过的模型都列出来：出现两个就说明走了备用，那又慢又贵，得看得见
    model_names = list(usage.get("by_model", {}).keys())
    if model_names:
        parts.append("模型 " + " + ".join(model_names))
    return "，".join(parts)


class UsageCollector(BaseCallbackHandler):
    """
    [WHY] 用量和耗时只存在于模型返回的那个对象里，而拿得到那一步的代码
          （common.invoke 取了 .content、analyze 的 with_structured_output 直接给解析结果）
          都把返回对象扔了。想在那些地方补记录，等于每个调用点都得改一遍，漏一个就静默少记。
    [FIX] 改成实现 LangChain 的回调钩子，在构造模型时挂上 —— 所有调用（普通 invoke、
          结构化输出、故障转移、重试）都会经过这里，调用点一个都不用动。
    [REF] LangChain BaseCallbackHandler 的回调协议
    """

    def __init__(self, model: str):
        """记住自己是谁 —— 用量要按模型分开记，才看得出主模型和备用的差别。"""
        self.model = model
        self._starts = {}  # run_id -> 开始时刻

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        """[WHY] on_llm_end 只收到结果、拿不到耗时，所以开始这一刻得自己记下来。"""
        # 按 run_id 存：一次问答里可能有好几个调用在跑
        self._starts[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        """[WHY] 用量就藏在 response 里，而这是外面唯一能顺手把它取走的地方。"""
        # 取走开始时刻：用完就删，免得 _starts 越堆越多
        started = self._starts.pop(run_id, None)
        seconds = time.perf_counter() - started if started else 0.0

        # response.generations 是二维的（外层一次调用、内层这次调用产生的候选），
        # 取第一个带用量的就够了
        usage = {}
        for gen_list in response.generations:
            for gen in gen_list:
                message = getattr(gen, "message", None)
                # 有的路（比如解析失败那次）不带用量，跳过
                if message is not None and getattr(message, "usage_metadata", None):
                    usage = message.usage_metadata

        add_llm_call(self.model, usage, seconds)
