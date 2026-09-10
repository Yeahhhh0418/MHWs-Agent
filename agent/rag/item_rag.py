"""
agent/rag/item_rag.py —— 物品图鉴 RAG（混合检索）
"""

import json

from langchain_core.documents import Document

from agent.rag.base import BaseRAG
from agent.logger import get_logger

logger = get_logger("rag.item")


# 物品类型中文映射
TYPE_CN_MAP = {
    "EXPENDABLE": "消耗品",
    "TOOL": "道具/工具",
    "MATERIAL": "素材/材料",
    "SHELL": "弹药/瓶",
    "GEM": "装饰品/宝珠",
    "POINT": "特产/点数物品",
    "BOTTLE": "瓶",
}


class ItemDataRAG(BaseRAG):
    _DB_NAME = "item"

    RAG_PROMPT_TEMPLATE = """你是一个专业的《怪物猎人：荒野》物品图鉴助手。请根据以下检索到的物品信息，回答玩家的问题。

【核心规则——务必遵守】：
玩家经常使用不精确的俗称或简称来描述物品，你必须根据检索结果自动匹配最接近的正式物品名。例如：

| 玩家可能的说法 | 实际指的游戏物品 | 推理逻辑 |
|--------------|----------------|--------|
| "回复药" | 【回复药】 | 直接匹配正式名称 |
| "大回复药" / "高级回复药" | 【回复药G】 | "大/高级"对应 G 级升级版 |
| "秘药" | 【秘药】 | 直接匹配 |
| "远古药" / "古代秘药" | 【远古秘药】 | "远古/古代"对应远古前缀 |
| "解毒的" / "解毒东西" | 【解毒药】 | 功能描述匹配 |
| "烤肉" / "烤熟的" | 【烤熟的肉】 | 俗称匹配 |
| "粉尘" / "生命粉" | 【生命粉尘】 | 简称匹配 |
| "陷阱" / "麻痹陷阱" | 【麻痹陷阱】 | 功能+类型匹配 |
| "捕获球" / "麻醉球" | 【捕获用麻醉球】 | 功能描述匹配 |
| "闪光弹" / "闪光" | 【闪光弹】 | 简称匹配 |
| "怪物的皮" / "龙皮" | 各种怪物皮素材 | 模糊描述匹配素材类 |

【物品类型说明】：
- 消耗品(EXPENDABLE)：回复药、肉、粉尘、果实等可使用的物品
- 道具/工具(TOOL)：陷阱、爆弹、投射器弹药等功能性道具
- 素材/材料(MATERIAL)：从怪物身上剥取的皮、鳞、爪等制作材料
- 弹药/瓶(SHELL)：弓弩用的各种弹药和瓶
- 装饰品/宝珠(GEM)：镶嵌在装备上的宝珠
- 特产/点数物品(POINT)：可用于交换点数的特产
- 瓶(BOTTLE)：可装备在弓上的瓶

【回答规范】：
1. 绝对不要说"没有找到你说的XX"或"攻略信息中没有XX"——只要检索结果里有名称或功能相似的物品，就直接告诉玩家。
2. 回答时先用一句话确认匹配，例如"你说的应该是【回复药G】"，然后给出物品详情。
3. 如果检索结果中有多个可能匹配的物品（如回复药、回复药G、远古秘药），把它们都列出来并说明区别（稀有度、效果、价格等）。
4. 回答应包含：物品名、类型、稀有度、效果/说明、售价/购买价。
5. 如果玩家问的是"做什么用的"或"怎么获得"，根据物品说明给出解释。
6. 只有在检索结果中确实完全没有任何相关物品时，才回答"抱歉，我的物品图鉴里暂时没有找到这个信息"。

【检索到的物品信息】：
{context}

【玩家的问题】：
{question}

【你的回答】："""

    def _load_data(self):
        """加载物品 JSON，转为 Document 列表。"""
        logger.info("加载物品数据", extra={"path": self._data_path})
        documents = []
        with open(self._data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        for item in data_list:
            name = item.get("name", "")
            if not name or name == "---":
                continue

            type_cn = TYPE_CN_MAP.get(item.get("type"), item.get("type", "未知"))
            rare = item.get("rare", "?")
            sell = item.get("sell_price", "?")
            buy = item.get("buy_price", "?")
            desc = item.get("description", "")

            content = (
                f"【{name}】是一类{type_cn}，稀有度 R{rare}。"
                f"说明：{desc}"
                f"售价 {sell}z / 购买价 {buy}z。"
            )

            documents.append(Document(page_content=content, metadata={
                "item_id": item.get("item_id"),
                "name": name,
                "type": item.get("type"),
                "type_cn": type_cn,
                "rare": rare,
                "sell_price": sell,
                "buy_price": buy,
                "description": desc,
            }))

        self._docs = documents
        logger.info("物品数据加载完成", extra={"count": len(documents)})