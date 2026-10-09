"""
src/app/logger.py —— 结构化日志。

控制台打带颜色的文本，同时往 log_file 里写一行一条的 JSON。
初始化只在 src/app/llm.py 里做一次（级别和路径都写死在那儿），其余地方直接用：

    from app.logger import get_logger

    logger = get_logger("nodes.route")     # 名字随意，会挂在 mhwi. 命名空间下
    logger.info("开始路由", extra={"question": "火龙怎么打"})

extra 里的字段会并进 JSON，也会追在控制台那行消息后面（长值截到 40 字符）。
"""

import contextvars
import json
import logging
import sys
import uuid
from datetime import datetime

import colorlog
from colorlog.escape_codes import escape_codes

# 当前正在处理的那次问答的编号。用 contextvars 而不是普通全局变量 ——
# 它按"执行上下文"隔离，以后上 Web（多请求并发）时每个请求各一份，不会串。
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


def new_request_id() -> str:
    """开一次问答时调用（str）：生成编号并写进当前执行上下文。"""
    # 8 位够用，短一点才好口述和 grep
    rid = uuid.uuid4().hex[:8]
    # 写进当前执行上下文，之后这个上下文里的日志自动都带上它
    _request_id.set(rid)
    return rid


def bind_request_id(rid: str) -> None:
    """
    [WHY] 不能只在起点设一次 —— LangGraph 给每个节点发独立的上下文副本
          （同一线程，但各节点是"兄弟"关系）。在 route 里 set 的值，
          retrieve / analyze / generate 一律读不到（实测过：同一线程 id，
          但下一个节点读出来是空）。
    [FIX] 所以每个图节点入口重新 set 一次，覆盖这个节点的整条调用栈
          —— 节点自己、以及它调用的 rag / common 那些日志就都带上了。
    """
    # 传空也要 set —— 不然会把上一个请求的编号漏进来
    _request_id.set(rid or "")


class JSONFormatter(logging.Formatter):
    """把每条日志格式化成一行 JSON，方便机器解析和以后排查。"""

    def format(self, record: logging.LogRecord) -> str:
        # 固定这几项打底
        log_entry = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # 如果有异常堆栈，也塞进日志里
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        # extra={...} 里塞的字段，原样并进 JSON
        if hasattr(record, "extra_fields") and record.extra_fields:
            log_entry.update(record.extra_fields)
        # ensure_ascii=False：中文原样写进去，不转成 \uXXXX
        return json.dumps(log_entry, ensure_ascii=False)


# 每个模块配一个颜色。原来所有模块名都是同一个紫色，route / retrieve / analyze /
# generate 在终端里混成一片，看不出哪一步到哪一步。
# 只影响控制台；JSON 落盘那份是纯文本，不带转义码。
_NAME_COLORS = {
    "nodes.route": "blue",
    "nodes.retrieve": "cyan",
    "nodes.analyze": "purple",
    "nodes.generate": "green",
    "chatllm": "yellow",
    # rag 那两层是噪音（只在首次加载时打），压成灰色退到背景
    "rag.base": "light_black",
    "rag.main": "light_black",
}


def _color_for(name: str) -> str:
    """按模块名取 ANSI 颜色码（str）；没配上的返回空串 = 终端默认色。"""
    color = _NAME_COLORS.get(name)
    if color is None:
        return ""
    return escape_codes[color]


class ConsoleFormatter(colorlog.ColoredFormatter):
    """
    [WHY] 原来控制台只打 message，extra 里的东西（查了哪个库、命中几条、为什么降级）
          一个都看不到 —— 想排查得去翻 JSON 文件，等于白打。
    [FIX] 把 extra 按 key=value 追在消息后面，长值截断；顺手去掉 mhwi. 前缀，
          模块名短一半；再按模块上色，让节点之间的边界一眼可见。
    """

    MAX_VALUE_LEN = 40

    def format(self, record: logging.LogRecord) -> str:
        # 同一条 record 会被多个 handler 共用（先控制台、后文件），
        # 所以名字改完必须还原 —— 否则 JSON 那边的 logger 字段会带上转义码
        original_name = record.name
        short_name = original_name.removeprefix("mhwi.")
        # 颜色只能加在名字里：colorlog 只按「级别」配色，配不了「模块」
        record.name = f"{_color_for(short_name)}{short_name}{escape_codes['reset']}"
        try:
            line = super().format(record)
        finally:
            record.name = original_name

        # 把 extra 拼成 key=value 追在消息后面
        extra = getattr(record, "extra_fields", None)
        if extra:
            parts = []
            for key, value in extra.items():
                text = str(value)
                # 长值截断 —— 一串制作材料能把整行撑爆
                if len(text) > self.MAX_VALUE_LEN:
                    text = text[: self.MAX_VALUE_LEN] + "…"
                parts.append(f"{key}={text}")
            line += "  " + " ".join(parts)
        return line


class _StructuredAdapter(logging.LoggerAdapter):
    """
    [WHY] logging 只认自己那套 extra（还得避开 LogRecord 已有的字段名），
          而我们想传的是任意业务字段。而且每处 logger.info 都手写一遍
          request_id 太啰嗦、容易漏。
    [FIX] 把调用方给的 extra 收进 extra_fields（JSONFormatter 认得它）；
          顺手把当前上下文的 request_id 也塞进去 —— 所以 route 那边设一次，
          之后所有节点的日志自动都带编号，不用逐个改。
    """

    def process(self, msg: str, kwargs: dict) -> tuple:
        # 拷一份，别动调用方传进来的那个字典
        extra = dict(kwargs.get("extra") or {})
        # 当前上下文有编号就自动带上，不用每处 logger.info 手写
        rid = _request_id.get()
        if rid:
            extra["request_id"] = rid
        # 包成 extra_fields：logging 只放行 LogRecord 上已存在的字段名
        if extra:
            kwargs["extra"] = {"extra_fields": extra}
        return msg, kwargs


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
) -> None:
    """初始化根 logger。全局调用一次即可（我们在 chatllm.py 里调用）。"""
    # 所有 logger 都挂在 mhwi 这个命名空间下
    root = logging.getLogger("mhwi")
    # 级别认不出来就退回 INFO，不崩
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # 先清干净：重复调用也不会把 handler 越挂越多
    root.handlers.clear()

    # 日志走 stderr —— stdout 留给答案，别混在一起
    console_handler = logging.StreamHandler(sys.stderr)

    # 定义彩色格式：时间(蓝) 等级(按级别配色) 模块名(按模块配色) 消息
    # 模块名那里不写死颜色 —— 由 ConsoleFormatter 按 _NAME_COLORS 自己往里塞
    log_colors = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }
    formatter_str = "%(log_color)s[%(asctime)s] %(levelname)-8s%(reset)s %(name)s  %(message)s"
    console_handler.setFormatter(
        ConsoleFormatter(
            formatter_str, datefmt="%H:%M:%S", log_colors=log_colors, reset=True, style="%"
        )
    )
    root.addHandler(console_handler)

    if log_file:
        from pathlib import Path

        # 目录不存在就先建出来
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        # 文件里依然保持 JSON 格式，方便以后机器读取
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)


def get_logger(name: str) -> _StructuredAdapter:
    """获取 mhwi 命名空间下的 logger，替代 print() 使用。"""
    # 统一挂在 mhwi. 下面，控制台显示时再去掉这个前缀
    raw_logger = logging.getLogger(f"mhwi.{name}")
    # 包一层适配器：extra 和 request_id 都在它里面处理
    return _StructuredAdapter(raw_logger, {})

def current_request_id() -> str:
    """取当前执行上下文里的问答编号（str）；没有就返回空串。"""
    return _request_id.get()