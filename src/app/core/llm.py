"""
src/app/core/llm.py —— 全局 LLM 实例。

模型名、接口地址、密钥一律从 .env 读；config.yaml 里只记"读哪个环境变量"。
公开的配置里不放任何服务商信息 —— 使用者接自己的模型就行。
"""

import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.core.config import get_config
from app.core.logger import get_logger, setup_logging
from app.core.metrics import UsageCollector

load_dotenv()

# 初始化结构化日志（整个程序只在这里调用一次）
setup_logging(level="INFO", log_file="logs/agent.log")

logger = get_logger("chatllm")

def _read_env(var_name: str, label: str) -> str:
    """
    [WHY] 模型名和接口地址要是写进 config.yaml，那就是公开的 —— 使用者要么照用别人的，
          要么得改仓库里的文件（改完还容易把自己的配置误提交上去）。
    [FIX] config 只记"读哪个环境变量"，值都放 .env（不进仓库）。取不到就说清去哪填，
          别等真去调模型时才报一个看不懂的 401。
    """
    value = (os.getenv(var_name) or "").strip()
    if not value:
        raise RuntimeError(
            f"{label}要的环境变量 {var_name} 没设（或为空）。"
            f"照着 .env.example 在项目根目录建一个 .env，填上你自己的模型信息。"
        )
    return value


def _build_single_llm(cfg: dict, label: str) -> BaseChatModel:
    """按一份模型配置建 LLM 实例（含重试）；模型名 / 地址 / 密钥都从 .env 读。"""
    # 三样都从环境变量取 —— config 里只有"读哪个变量"
    model = _read_env(cfg["model_env"], label)
    base_url = _read_env(cfg["base_url_env"], label)
    api_key = _read_env(cfg["api_key_env"], label)

    # 走 OpenAI 兼容协议：换一家服务商只改 .env，代码和配置都不用动
    kwargs = {
        "model": model,
        "model_provider": "openai",
        "base_url": base_url,
        "api_key": api_key,
        "temperature": cfg.get("temperature", 0.7),
        "max_retries": cfg.get("max_retries", 3),  # ← 第1层
        "timeout": cfg.get("timeout", 30),
        # 挂用量收集器：所有模型调用自动被记账，调用点不用改
        "callbacks": [UsageCollector(model)],
    }
    return init_chat_model(**kwargs)


@lru_cache(maxsize=1)
def get_llms() -> tuple[BaseChatModel, ...]:
    """
    [WHY] 原来用 primary.with_fallbacks(...)，它是个黑盒 —— 备用模型答了你也看不出来，
          而主/备用的质量差得远（实测备用模型连"这是怪物猎人"都没意识到）。
    [FIX] 干脆返回 [主模型, 备用…] 的列表，由 invoke_with_fallback 去遍历，
          每次切换都能记日志。
    """
    # 读 llm 那一段配置
    llm_cfg = get_config()["llm"]

    # 主模型排第一个 —— 兜底顺序就是靠这个列表顺序
    models = [_build_single_llm(llm_cfg["primary"], "主模型")]
    logger.info("主模型加载完成", extra={"model": model_name(models[0])})

    # 备用是可选的：.env 里没填模型名就等于不用（不做一长串链，有多家需求再扩）
    fallback_cfg = llm_cfg.get("fallback")
    if fallback_cfg and (os.getenv(fallback_cfg["model_env"]) or "").strip():
        models.append(_build_single_llm(fallback_cfg, "备用模型"))
        logger.info(
            "已配置 1 个备用模型",
            extra={"models": [model_name(m) for m in models[1:]]},
        )
    else:
        logger.info("没配备用模型，只用主模型")

    return tuple(models)


def model_name(llm: BaseChatModel) -> str:
    """取模型名，用来记日志（str）。"""
    return getattr(llm, "model_name", type(llm).__name__)


def invoke_with_fallback(
    call: Callable[[BaseChatModel], Any], no_fallback_on: tuple = ()
) -> tuple[Any, str]:
    """
    [WHY] with_fallbacks 换模型时一声不吭 —— 你拿到一个备用模型答的答案，
          还以为主模型好好的。
    [FIX] 按 [主, 备用…] 顺序试 call(llm)：每次切换记一条 warning，成功时返回
          (结果, 用了哪个模型)。
          no_fallback_on 里的异常直接抛、不换模型 —— 换模型救不了那类错
          （比如输出格式不合格，换个模型一样不合格，该走回喂重试）。
    """
    # 记住第一个错，全挂了的时候抛它
    first_error = None
    # 按 [主, 备用…] 的顺序挨个试
    for index, llm in enumerate(get_llms()):
        try:
            result = call(llm)
        # 这类错换模型也救不了，直接往外抛
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

    # 全试完了还是不行 —— `first_error or e` 保住的是第一个错，不是最后一个
    raise first_error


logger.info(
    "LLM 加载完成",
    extra={"model": os.getenv(get_config()["llm"]["primary"]["model_env"], "")},
)

if __name__ == "__main__":
    # 测试 LLM 是否能正常工作（会真的调模型）
    text, used = invoke_with_fallback(lambda llm: llm.invoke("你好，世界！").content)
    print(f"[{used}] {text}")
