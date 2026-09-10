"""
agent/rag/equipment_rag.py —— 装备制作 RAG（混合检索）
"""

import json

from langchain_core.documents import Document

from agent.rag.base import BaseRAG

from agent.logger import get_logger

logger = get_logger("rag.equipment")


class EquipmentCraftingRAG(BaseRAG):
    _DB_NAME = "equipment"

    # 俗称部位词 → 正式词。检索前会把这些正式词追加到查询里，
    # 弥补"手→腕甲""衣服→铠甲"这类俗称映射缺口（否则 BM25/向量都命中不了）。
    QUERY_SYNONYMS = {
        "头": "头盔 头部",
        "衣服": "铠甲 胸部",
        "手": "腕甲 腕部",
        "腰": "腰甲 腰部",
        "腿": "护腿 腿部",
    }

    RAG_PROMPT_TEMPLATE = """你是一个专业的《怪物猎人：荒野》攻略助手。请根据以下检索到的攻略信息，回答玩家的问题。

【核心规则——务必遵守】：
玩家经常使用不精确的俗称来描述装备，你必须根据检索结果自动匹配最接近的正式装备名。例如：

| 玩家可能的说法 | 实际指的游戏装备 | 推理逻辑 |
|--------------|----------------|--------|
| "炎尾龙的头" | 【炎尾龙头盔】 | "炎尾龙"是怪物名，"头"对应头部防具→头盔 |
| "缠蛙的衣服" | 【缠蛙铠甲】 | "缠蛙"是怪物名，"衣服"对应胸部防具→铠甲 |
| "辟兽的手" | 【辟兽腕甲】 | "辟兽"是怪物名，"手"对应腕部防具→腕甲 |
| "沙海龙的腰" | 【沙海龙腰甲】 | "沙海龙"是怪物名，"腰"对应腰部防具→腰甲 |
| "沼喷龙的腿" | 【沼喷龙护腿】 | "沼喷龙"是怪物名，"腿"对应腿部防具→护腿 |
| "护火龙的胸" | 【护火龙铠甲】 | "护火龙"是怪物名，"胸"对应胸部防具→铠甲 |
| "希望面具" | 【希望面具】 | 玩家直接说出完整装备名，直接匹配 |

【回答规范】：
1. 绝对不要说"没有找到你说的XX"或"攻略信息中没有XX"——只要检索结果里有名称相似的装备，就直接告诉玩家它叫什么、怎么制作。
2. 回答时先用一句话确认匹配，例如"根据你的描述，对应的装备是【炎尾龙头盔】"，然后给出制作信息。
3. 如果检索结果中有多个可能匹配的装备（如炎尾龙头盔、炎尾龙头盔α、炎尾龙头盔β），把它们都列出来并说明区别。
4. 如果检索结果中同时包含同名武器和防具，优先列出防具再列武器（或反过来），让玩家选择。
5. 只有在检索结果中确实完全没有任何相关装备时，才回答"抱歉，我的狩猎笔记里暂时没有找到这个信息"。

【检索到的攻略信息】：
{context}

【玩家的问题】：
{question}

【你的回答】："""

    def _load_data(self):
        """加载装备 JSON，转为 Document 列表。"""
        logger.info("加载装备数据", extra={"path": self._data_path})
        documents = []
        with open(self._data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        for data in data_list:
            meta = data.get("metadata", {})
            # ChromaDB 不接受空列表，把空列表转成 None
            sanitized = {}
            for k, v in meta.items():
                if isinstance(v, list) and len(v) == 0:
                    sanitized[k] = None
                else:
                    sanitized[k] = v

            documents.append(Document(
                page_content=data["content"],
                metadata=sanitized,
            ))

        self._docs = documents
        logger.info("装备数据加载完成", extra={"count": len(documents)})