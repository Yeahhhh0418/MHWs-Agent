"""
agent/rag/monster_rag.py —— 怪物图鉴 RAG（混合检索）
涵盖大型怪物(Monster)和环境生物(EnvironmentCreature)两类。
"""

import json

from langchain_core.documents import Document

from agent.rag.base import BaseRAG
from agent.logger import get_logger

logger = get_logger("rag.monster")


class MonsterDataRAG(BaseRAG):
    _DB_NAME = "monster"

    RAG_PROMPT_TEMPLATE = """你是一个专业的《怪物猎人：荒野》怪物图鉴与狩猎顾问。请根据以下检索到的怪物/生物信息，回答玩家的问题。

【核心规则——务必遵守】：
玩家经常使用不精确的俗称、简称或外号来描述怪物，你必须根据检索结果自动匹配最接近的正式名称。例如：

| 玩家可能的说法 | 实际指的游戏怪物 | 推理逻辑 |
|--------------|----------------|--------|
| "雌火" / "雌火龙" | 【雌火龙】 | 简称/全称匹配 |
| "火龙" / "雄火" | 【火龙】 | 常用称呼匹配 |
| "丝瓜" / "恐暴" | 【恐暴龙】 | 外号/俗称匹配 |
| "咩咩子" / "灭尽" | 【灭尽龙】 | 玩家社区昵称匹配 |
| "太太" / "炎妃" | 【炎妃龙】 | 昵称/简称匹配 |
| "警察" / "金狮子" | 【金狮子】 | 外号匹配 |
| "角龙" / "黑角" | 【角龙】或【黑角龙】 | 简称匹配 |
| "烤鱼" / "熔岩龙" | 【熔岩龙】 | 外号匹配 |
| "青蛙" / "瓦斯蛙" | 【麻痹瓦斯蛙】 | 功能+外形匹配（环境生物） |
| "闪光虫" / "发光虫" | 【闪光翅虫】 | 功能描述匹配（环境生物） |

【数据类别说明】：
检索结果中可能包含两类数据：
- **Monster（大型怪物）**：拥有狩猎攻略的怪物，包含种类、生态、弱点、招式应对等信息。
- **EnvironmentCreature（环境生物）**：小型生态生物，包含外形特征和生态习性等信息。

【回答规范】：
1. 绝对不要说"没有找到你说的XX"或"图鉴中没有XX"——只要检索结果里有名称或特征相似的怪物/生物，就直接告诉玩家。
2. 回答时先用一句话确认匹配，例如"根据你的描述，对应的应该是【雌火龙】"。
3. 如果检索结果中有多个可能匹配的怪物（如雌火龙、火龙、樱火龙），把它们都列出来并说明区别。
4. 如果玩家问的是**打法/攻略**：先简要介绍怪物种类和特征，再给出具体的狩猎攻略（弱点、招式应对、部位破坏建议）。
5. 如果玩家问的是**生态/背景**：优先用生态详情回答，再补充特征概述。
6. 如果玩家问的是**环境生物**：简要描述其外形和功能。
7. 如果怪物有历战个体、历战王个体、特殊个体或狂龙化形态，在回答末尾提醒玩家存在这些变体。
8. 只有在检索结果中确实完全没有任何相关怪物/生物时，才回答"抱歉，我的狩猎笔记里暂时没有找到这个信息"。

【检索到的怪物/生物信息】：
{context}

【玩家的问题】：
{question}

【你的回答】："""

    def _load_data(self):
        """加载怪物 JSON，转为 Document 列表。"""
        logger.info("加载怪物数据", extra={"path": self._data_path})
        documents = []
        with open(self._data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        for entry in data_list:
            name = entry.get("name", "")
            if not name or name == "---":
                continue

            category = entry.get("data_category", "")
            if category == "Monster":
                content = self._build_monster_content(entry)
            elif category == "EnvironmentCreature":
                content = self._build_env_creature_content(entry)
            else:
                continue

            if not content.strip():
                continue

            metadata = {
                "name": name,
                "data_category": category,
                "category_type": entry.get("category_type"),
                "short_intro": entry.get("short_intro"),
                "ecology": entry.get("ecology"),
                "hunting_guide": entry.get("hunting_guide"),
                "variant_name": entry.get("variant_name"),
                "variant_intro": entry.get("variant_intro"),
                "frenzy_name": entry.get("frenzy_name"),
                "tempered_name": entry.get("tempered_name"),
                "tempered_king_name": entry.get("tempered_king_name"),
                "env_short_desc": entry.get("env_short_desc"),
                "env_long_desc": entry.get("env_long_desc"),
            }
            # 去掉 metadata 里的 None 值（Chroma 不接受 None）
            metadata = {k: v for k, v in metadata.items() if v is not None}

            documents.append(Document(page_content=content, metadata=metadata))

        self._docs = documents
        monster_count = sum(1 for d in documents if d.metadata.get("data_category") == "Monster")
        env_count = sum(1 for d in documents if d.metadata.get("data_category") == "EnvironmentCreature")
        logger.info("怪物数据加载完成", extra={"count": len(documents), "monster": monster_count, "env": env_count})

    @staticmethod
    def _build_monster_content(entry: dict) -> str:
        parts = [f"【{entry.get('name', '')}】是一种{entry.get('category_type', '')}大型怪物。"]
        if entry.get("short_intro"):
            parts.append(f"特征概述：{entry['short_intro']}")
        if entry.get("ecology"):
            parts.append(f"生态详情：{entry['ecology']}")
        if entry.get("hunting_guide"):
            parts.append(f"狩猎攻略：{entry['hunting_guide']}")
        if entry.get("variant_name"):
            extra = f"（{entry['variant_intro']}）" if entry.get("variant_intro") else ""
            parts.append(f"特殊个体：{entry['variant_name']}{extra}")
        if entry.get("frenzy_name"):
            parts.append(f"狂龙化形态：{entry['frenzy_name']}")
        if entry.get("tempered_name"):
            parts.append(f"历战个体：{entry['tempered_name']}")
        if entry.get("tempered_king_name"):
            parts.append(f"历战王个体：{entry['tempered_king_name']}")
        return "。".join(parts) + "。"

    @staticmethod
    def _build_env_creature_content(entry: dict) -> str:
        parts = [f"【{entry.get('name', '')}】是一种环境生物。"]
        if entry.get("short_intro"):
            parts.append(f"特征概述：{entry['short_intro']}")
        if entry.get("env_short_desc"):
            parts.append(f"简述：{entry['env_short_desc']}")
        if entry.get("env_long_desc"):
            parts.append(f"生态详情：{entry['env_long_desc']}")
        return "。".join(parts) + "。"