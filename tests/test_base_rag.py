import pytest

from agent.rag.base import BaseRAG


class _DummyRAG(BaseRAG):
    _DB_NAME = "item"  # 复用 config 里已有的 item 配置

    def _load_data(self):
        self._docs = []

    # 下面三个方法覆写掉，避免真的建 Chroma / 组链
    def _init_vectorstore(self):
        self._vectorstore = None

    def _init_retriever(self):
        self._retriever = None

    def _init_chain(self):
        self._chain = None


def _make():
    return _DummyRAG()


def test_query_before_setup_raises():
    rag = _make()
    with pytest.raises(RuntimeError):
        rag.query("hi")


def test_setup_sets_ready():
    rag = _make()
    rag.setup()
    assert rag.is_ready is True
