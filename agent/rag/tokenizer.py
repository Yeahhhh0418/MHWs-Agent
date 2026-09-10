"""
agent/rag/tokenizer.py —— 中文字符级分词器
为什么需要它：BM25 默认按空格分词，对中文无效（中文没空格）。
这个函数把每个汉字拆成独立 token，英文/数字保持连续，
确保"锁刃龙"这类关键词检索不会漏。
"""


def chinese_char_tokenizer(text: str) -> list[str]:
    tokens = []
    buf = ""
    for ch in text:
        if "一" <= ch <= "鿿":  # 是汉字
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
        elif ch.isalnum():  # 英文或数字，攒在一起
            buf += ch
        else:  # 标点等分隔符，把攒的英文/数字先落盘
            if buf:
                tokens.append(buf)
                buf = ""
    if buf:
        tokens.append(buf)
    return [t for t in tokens if t]