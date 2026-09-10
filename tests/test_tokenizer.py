from agent.rag.tokenizer import chinese_char_tokenizer


def test_pure_chinese():
    assert chinese_char_tokenizer("雌火龙") == ["雌", "火", "龙"]


def test_mixed_chinese_and_letter():
    assert chinese_char_tokenizer("回复药G") == ["回", "复", "药", "G"]


def test_pure_english():
    assert chinese_char_tokenizer("Alpha") == ["Alpha"]


def test_empty_string():
    assert chinese_char_tokenizer("") == []


def test_with_punctuation():
    assert chinese_char_tokenizer("锁刃龙的头") == ["锁", "刃", "龙", "的", "头"]
