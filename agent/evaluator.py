"""
agent/evaluator.py —— 两个轻量回答质量指标（不依赖 LLM）
"""

import re


def check_empty_answer(content: str) -> bool:
    """判断回答是否为空，或只是简短的致歉。"""
    if not content or not content.strip():
        return True
    apology_words = ["抱歉", "没有找到", "暂时没有", "无法回答"]
    stripped = content.strip()
    if len(stripped) < 10 and any(w in stripped for w in apology_words):
        return True
    return False


def extract_entities(text: str) -> list[str]:
    """提取文本里所有【...】括起来的实体名。"""
    return re.findall(r"【([^】]+)】", text)


def check_named_entity_overlap(answer: str, retrieved_docs: list[str]) -> float:
    """回答里【实体】有多大比例能在检索文档中找到。

    返回 0.0 ~ 1.0。回答里没有任何【】实体时返回 1.0（跳过检测）。
    """
    answer_entities = extract_entities(answer)
    if not answer_entities:
        return 1.0

    doc_text = "\n".join(retrieved_docs)
    doc_entities = set(extract_entities(doc_text))
    if not doc_entities:
        return 0.0

    hit = sum(1 for e in answer_entities if e in doc_entities)
    return hit / len(answer_entities)