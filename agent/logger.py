"""
agent/logger.py —— 结构化日志模块
用法：
    from agent.logger import setup_logging, get_logger
    setup_logging(level="INFO", fmt="json", log_file="logs/agent.log")
    logger = get_logger(__name__)
    logger.info("路由完成", extra={"first_db": "equipment", "keyword": "炎尾龙"})
"""

import logging
import sys
import json
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """把每条日志格式化成一行 JSON，方便机器解析和以后排查。"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
        return json.dumps(log_entry, ensure_ascii=False)


class _StructuredAdapter(logging.LoggerAdapter):
    """让 logger.info("msg", extra={...}) 里的 extra 字段能进到 JSON 里。"""

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        if extra:
            kwargs["extra"] = {"extra_fields": extra}
        return msg, kwargs


def setup_logging(
    level: str = "INFO",
    fmt: str = "json",
    log_file: Optional[str] = None,
) -> None:
    """初始化根 logger。全局调用一次即可（我们在 chatllm.py 里调用）。"""
    root = logging.getLogger("mhwi")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    root.addHandler(console_handler)

    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(file_handler)


def get_logger(name: str):
    """获取 mhwi 命名空间下的 logger，替代 print() 使用。"""
    raw_logger = logging.getLogger(f"mhwi.{name}")
    return _StructuredAdapter(raw_logger, {})