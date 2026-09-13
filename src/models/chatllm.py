"""
agent/chatllm.py —— 全局 LLM 实例
模型名从 config.yaml 读取，不再硬编码。
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from mhwi_agent.config import get_config
from mhwi_agent.logger import setup_logging, get_logger

load_dotenv()

# 初始化结构化日志（整个程序只在这里调用一次）
setup_logging(level="INFO", log_file="logs/agent.log")

logger = get_logger("chatllm")

# 从 config.yaml 读取模型名，初始化大模型
llm = init_chat_model(model=get_config()["llm"]["model"])
logger.info("LLM 加载完成", extra={"model": get_config()["llm"]["model"]})
