"""
agent/retry.py —— 两个工具：安全解析 JSON + 带重试的调用
"""

import json
import re
import time


def safe_json_parse(raw_text, required_keys=None, fallback=None):
    """多层恢复策略解析 LLM 的 JSON 输出，从不抛异常。

    解析顺序：
      1) 直接 json.loads()
      2) 提取 ```json ... ``` 代码块
      3) 正则提取最外层 {...}
    都失败（或缺 required_keys）时，返回 fallback。

    参数：
        raw_text      LLM 原始输出（可能混有文字/markdown）
        required_keys 必须包含的键，缺了就返回 fallback
        fallback      解析失败时返回的保底字典
    """
    required_keys = required_keys or []
    fallback = fallback or {}

    def _try(text):
        return json.loads(text)

    result = None

    # 策略 1：直接解析
    try:
        result = _try(raw_text)
    except (json.JSONDecodeError, TypeError):
        result = None

    # 策略 2：提取 markdown 代码块里的 JSON
    if result is None:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
        if m:
            try:
                result = _try(m.group(1))
            except (json.JSONDecodeError, TypeError):
                result = None

    # 策略 3：提取最外层花括号 {...}
    if result is None:
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if m:
            try:
                result = _try(m.group(0))
            except (json.JSONDecodeError, TypeError):
                result = None

    # 校验：解析结果必须是 dict，且包含所有必需键
    if result is None or not isinstance(result, dict):
        return dict(fallback)
    if required_keys and not all(k in result for k in required_keys):
        return dict(fallback)
    return result


def retry_call(fn, max_retries=2, base_delay=1.0):
    """调用 fn()，失败时用指数退避重试。总尝试次数 = 1 + max_retries。

    参数：
        fn          一个不接受参数的函数（返回要调用的结果）
        max_retries 最多重试几次
        base_delay  第一次重试前的等待秒数（之后每次翻倍）
    """
    delay = base_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries:
                raise  # 最后一次也失败，就把异常抛给上层去降级
            time.sleep(delay)
            delay *= 2  # 指数退避