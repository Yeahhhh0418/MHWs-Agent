"""
src/app/llm.py —— 全局 LLM 实例
模型名从 config.yaml 读取，不再硬编码。
"""

import os as _os
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.config import get_config
from app.logger import get_logger, setup_logging

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
        "timeout": cfg.get("timeout", 30),
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
def get_llms() -> tuple[BaseChatModel, ...]:
    """
    [WHY] 原来用 primary.with_fallbacks(...)，它是个黑盒 —— 备用模型答了你也看不出来，
          而主/备用的质量差得远（实测备用模型连"这是怪物猎人"都没意识到）。
    [FIX] 干脆返回 [主模型, 备用…] 的列表，由 invoke_with_fallback 去遍历，
          每次切换都能记日志。
    """
    llm_cfg = get_config()["llm"]

    models = [_build_single_llm(llm_cfg["primary"])]
    logger.info("主模型加载完成", extra={"model": llm_cfg["primary"]["model"]})

    for cfg in llm_cfg.get("fallbacks", []):
        models.append(_build_single_llm(cfg))
    if len(models) > 1:
        logger.info(
            "已配置 %d 个备用模型",
            len(models) - 1,
            extra={"models": [model_name(m) for m in models[1:]]},
        )
    return tuple(models)


def model_name(llm: BaseChatModel) -> str:
    """取模型名，用来记日志（str）。"""
    return getattr(llm, "model_name", type(llm).__name__)


def invoke_with_fallback(call, no_fallback_on: tuple = ()) -> tuple:
    """
    [WHY] with_fallbacks 换模型时一声不吭 —— 你拿到一个备用模型答的答案，
          还以为主模型好好的。
    [FIX] 按 [主, 备用…] 顺序试 call(llm)：每次切换记一条 warning，成功时返回
          (结果, 用了哪个模型)。
          no_fallback_on 里的异常直接抛、不换模型 —— 换模型救不了那类错
          （比如输出格式不合格，换个模型一样不合格，该走回喂重试）。
    """
    first_error = None
    for index, llm in enumerate(get_llms()):
        try:
            result = call(llm)
        except no_fallback_on:
            raise
        except Exception as e:  # noqa: BLE001 —— 抓全是对的：这个函数的职责就是"谁挂都换下一个"
            first_error = first_error or e
            logger.warning(
                f"模型调用失败，换成备用模型{model_name(llm)}",
                extra={"model": model_name(llm), "error": f"{type(e).__name__}: {str(e)[:100]}"},
            )
            continue

        if index > 0:
            # 走到这儿说明主模型挂了 —— 这条必须看得见，不然你会以为答案是主模型给的
            logger.warning(f"这次是备用模型{model_name(llm)}答的", extra={"model": model_name(llm)})
        return result, model_name(llm)

    raise first_error


logger.info("LLM 加载完成", extra={"model": get_config()["llm"]["primary"]["model"]})

if __name__ == "__main__":
    # 测试 LLM 是否能正常工作（会真的调模型）
    text, used = invoke_with_fallback(lambda llm: llm.invoke("你好，世界！").content)
    print(f"[{used}] {text}")
