import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage

from agent.nodes import route_node, retrieve_node, analyze_node, generate_node


def _state(**overrides):
    s = {
        "messages": [],
        "user_question": "测试问题",
        "current_db": "equipment",
        "query_keyword": "测试关键词",
        "collected_info": [],
        "next_action": "finish",
        "iteration_count": 0,
    }
    s.update(overrides)
    return s


# ── route_node ───────────────────────────────────────────
def test_route_node_normal():
    state = {"messages": [HumanMessage(content="怎么做装备")]}
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(
        content='{"first_db": "equipment", "first_keyword": "装备"}'
    )
    with patch("agent.nodes.llm", fake_llm):
        result = route_node(state)
    assert result["current_db"] == "equipment"
    assert result["query_keyword"] == "装备"


def test_route_node_degraded():
    state = {"messages": [HumanMessage(content="hi")]}
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = Exception("boom")
    with patch("agent.nodes.llm", fake_llm), \
         patch("agent.retry.time.sleep", return_value=None):
        result = route_node(state)
    assert result["current_db"] == "equipment"
    assert result["query_keyword"] == "hi"


# ── retrieve_node ────────────────────────────────────────
def test_retrieve_node_normal():
    state = _state()
    fake_rag = MagicMock()
    fake_rag.query.return_value = "检索到的内容"
    with patch("agent.nodes._get_rag", return_value=fake_rag):
        result = retrieve_node(state)
    assert result["collected_info"][0]["content"] == "检索到的内容"
    assert result["iteration_count"] == 1


def test_retrieve_node_degraded():
    state = _state()
    with patch("agent.nodes._get_rag", side_effect=Exception("boom")):
        result = retrieve_node(state)
    assert "检索失败" in result["collected_info"][0]["content"]


# ── analyze_node ─────────────────────────────────────────
def test_analyze_node_finish():
    state = _state(collected_info=[])
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(content='{"next_action": "finish"}')
    with patch("agent.nodes.llm", fake_llm):
        result = analyze_node(state)
    assert result["next_action"] == "finish"


def test_analyze_node_degraded():
    state = _state(collected_info=[])
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = Exception("boom")
    with patch("agent.nodes.llm", fake_llm), \
         patch("agent.retry.time.sleep", return_value=None):
        result = analyze_node(state)
    assert result["next_action"] == "finish"


# ── generate_node ────────────────────────────────────────
def test_generate_node_normal():
    state = _state(collected_info=[{"db": "equipment", "keyword": "x", "content": "内容"}])
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(content="正常回答")
    with patch("agent.nodes.llm", fake_llm):
        result = generate_node(state)
    assert result["final_answer"] == "正常回答"


def test_generate_node_degraded():
    state = _state(collected_info=[{"db": "equipment", "keyword": "x", "content": "内容"}])
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = Exception("boom")
    with patch("agent.nodes.llm", fake_llm), \
         patch("agent.retry.time.sleep", return_value=None):
        result = generate_node(state)
    assert result["final_answer"]  # 非空（降级为检索内容）
