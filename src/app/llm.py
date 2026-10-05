"""
src/app/llm.py —— 全局 LLM 实例
模型名从 config.yaml 读取，不再硬编码。
"""

import os as _os
from dotenv import load_dotenv
from functools import lru_cache
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from app.config import get_config
from app.logger import setup_logging, get_logger

load_dotenv()

# 初始化结构化日志（整个程序只在这里调用一次）
setup_logging(level="INFO", log_file="logs/agent.log")

logger = get_logger("chatllm")

# 从 config.yaml 读取模型名，初始化大模型

def _build_single_llm(cfg: dict) -> BaseChatModel:
    """根据单个模型配置构建 LLM 实例（含重试）"""
    kwargs = {
        "model": cfg["model"],
        "temperature": cfg.get("temperature", 0.7),
        "max_retries": cfg.get("max_retries", 3),  # ← 第1层
    }
    if base_url := cfg.get("base_url"):
        kwargs["model_provider"] = "openai"
        kwargs["base_url"] = base_url
        # 走了 openai 兼容协议，LangChain 只认 OPENAI_API_KEY；
        # 各家密钥存在 .env 里的名字不同，所以由 config 指定变量名。
        api_key_env = cfg.get("api_key_env")
        if not api_key_env:
            raise ValueError(f"模型 {cfg['model']} 配了 base_url 但没配 api_key_env")
        if not (api_key := _os.getenv(api_key_env)):
            raise RuntimeError(f"环境变量 {api_key_env} 未设置，请检查 .env")
        kwargs["api_key"] = api_key
    else:
        kwargs["model_provider"] = cfg.get("provider", "openai")

    return init_chat_model(**kwargs)

@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    获取带完整防御链的 LLM 实例。
    主模型 → 备用模型1 → 备用模型2 → 全部失败时由调用方捕获异常
    """
    llm_cfg = get_config()["llm"]
    
    # 构建主模型
    primary = _build_single_llm(llm_cfg["primary"])
    logger.info("主模型加载完成", extra={"model": llm_cfg["primary"]["model"]})
    
    # 构建备用模型链（第2层）
    fallback_configs = llm_cfg.get("fallbacks", [])
    if fallback_configs:
        fallback_llms = [_build_single_llm(cfg) for cfg in fallback_configs]

        # with_fallbacks 是 LangChain 原生方法，自动在主模型抛异常时切换
        primary = primary.with_fallbacks(fallback_llms)

        logger.info(
            "已配置 %d 个备用模型", 
            len(fallback_llms),
            extra={"models": [c["model"] for c in fallback_configs]}
        )
    return primary  

logger.info("LLM 加载完成", extra={"model": get_config()["llm"]["primary"]["model"]})

if __name__ == "__main__":
    # 测试 LLM 是否能正常工作
    llm = get_llm()
    response = llm.invoke("你好，世界！")
    print(response.content)