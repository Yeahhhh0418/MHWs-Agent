"""
tests/conftest.py —— pytest 的公共配置
"""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_llm():
    """返回一个假的 LLM：invoke() 的行为由测试自己设定。"""
    return MagicMock()
