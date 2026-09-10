import pytest

from agent.retry import safe_json_parse, retry_call


def test_normal_json():
    assert safe_json_parse('{"a": 1}') == {"a": 1}


def test_markdown_code_block():
    raw = '```json\n{"first_db": "equipment", "first_keyword": "x"}\n```'
    assert safe_json_parse(raw)["first_db"] == "equipment"


def test_naked_braces():
    raw = '前缀文字 {"first_db": "item", "first_keyword": "y"} 后缀'
    assert safe_json_parse(raw)["first_db"] == "item"


def test_garbage_returns_fallback():
    assert safe_json_parse("完全乱码", fallback={"a": 1}) == {"a": 1}


def test_missing_required_keys():
    raw = '{"foo": "bar"}'
    assert safe_json_parse(raw, required_keys=["first_db"], fallback={"x": 0}) == {"x": 0}


def test_retry_eventually_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("抖动")
        return "ok"

    assert retry_call(flaky, max_retries=3, base_delay=0.0) == "ok"
    assert calls["n"] == 3


def test_retry_gives_up():
    def always_fail():
        raise ValueError("坏")

    with pytest.raises(ValueError):
        retry_call(always_fail, max_retries=1, base_delay=0.0)
