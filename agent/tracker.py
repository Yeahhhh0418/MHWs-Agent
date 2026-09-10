"""
agent/tracker.py —— 轻量级调用追踪
记录一次 graph.invoke() 里每个节点的耗时、输入输出、错误。
"""

import time

from agent.logger import get_logger

logger = get_logger("tracker")

# 全局当前 tracker（通过 set/get 传递，避免改 LangGraph 的 state）
_tracker = None


def set_tracker(t):
    global _tracker
    _tracker = t


def get_tracker():
    return _tracker


def _truncate(obj, max_len=200):
    """截断过长的字符串/结构，避免日志爆炸。"""
    if isinstance(obj, str):
        return obj[:max_len] + ("..." if len(obj) > max_len else "")
    if isinstance(obj, dict):
        return {k: _truncate(v, max_len) for k, v in list(obj.items())[:20]}
    if isinstance(obj, list):
        return [_truncate(v, max_len) for v in obj[:20]]
    # LangChain 的 HumanMessage/AIMessage 等消息对象：只取 .content 字符串，
    # 否则 json.dumps 会抛 "Object of type HumanMessage is not JSON serializable"
    if hasattr(obj, "content") and isinstance(obj.content, str):
        return _truncate(obj.content, max_len)
    return obj


class TraceRecord:
    """单次节点调用的追踪记录。"""

    def __init__(self, node_name):
        self.node_name = node_name
        self.start_time = time.time()
        self.duration = None
        self.error = None
        self.input_snapshot = None
        self.output_snapshot = None

    def finish(self, output, error=None):
        self.duration = time.time() - self.start_time
        self.output_snapshot = _truncate(output)
        self.error = error

    def to_dict(self):
        return {
            "node": self.node_name,
            "duration": round(self.duration, 4) if self.duration is not None else None,
            "error": self.error,
            "input": self.input_snapshot,
            "output": self.output_snapshot,
        }


class AgentTracker:
    """一次 graph.invoke() 的完整追踪器。"""

    def __init__(self):
        self.records = []

    def start_node(self, node_name, state_snapshot):
        rec = TraceRecord(node_name)
        rec.input_snapshot = _truncate(state_snapshot)
        self.records.append(rec)
        return rec

    def end_node(self, rec, output, error=None):
        rec.finish(output, error)

    def flush_to_log(self):
        """把所有记录写入日志。即使中途崩溃也会被调用（见 graph.py）。"""
        for rec in self.records:
            logger.info("节点追踪", extra=rec.to_dict())

    def summarize(self) -> str:
        """返回人类可读的节点耗时摘要，方便在终端里一眼看明白。

        直接回答 Phase 4 自检的四个问题：几轮迭代、每轮用什么库/关键词、
        最慢的节点是哪个。
        """
        lines = []
        n_retrieve = 0
        max_dur = 0.0
        slowest = None
        for rec in self.records:
            dur = rec.duration or 0.0
            if rec.duration is not None and dur > max_dur:
                max_dur = dur
                slowest = rec.node_name
            dur_s = f"{dur:.3f}s" if rec.duration is not None else "未结束"
            err = f"，error={rec.error}" if rec.error else ""
            extra = ""
            # 检索节点最能说明"这一轮查了哪个库、什么关键词"
            if rec.node_name == "retrieve" and isinstance(rec.input_snapshot, dict):
                db = rec.input_snapshot.get("current_db", "?")
                kw = rec.input_snapshot.get("query_keyword", "?")
                extra = f"   ({db} / {kw})"
                n_retrieve += 1
            lines.append(f"  {rec.node_name}: {dur_s}{extra}{err}")

        head = f"共 {len(self.records)} 个节点调用，检索 {n_retrieve} 轮"
        if slowest:
            head += f"，最慢节点: {slowest} ({max_dur:.3f}s)"
        return head + "\n" + "\n".join(lines)


def track_node(node_name, fn, state):
    """包裹一个节点函数：自动记录开始/结束/错误。"""
    tracker = get_tracker()
    rec = tracker.start_node(node_name, dict(state)) if tracker else None
    try:
        result = fn(state)
        if rec:
            tracker.end_node(rec, result)
        return result
    except Exception as e:
        if rec:
            tracker.end_node(rec, {}, error=str(e))
        raise

