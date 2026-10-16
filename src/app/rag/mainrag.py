"""
四个库（装备 / 武器 / 怪物 / 素材）的 RAG 实现。
共同流程都在 rag.base.BaseRAG 里，这里只解决「各自的数据怎么读成 Document」。
"""

from pathlib import Path
from typing import ClassVar

from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.core.logger import get_logger
from app.rag.base import BaseRAG

logger = get_logger("rag.main")


class EquipmentCraftingRAG(BaseRAG):
    """
    [WHY] 装备表是英文 key，parts_type 还存着 HELM/BODY 这类枚举，直接拼进正文既会被玩家搜到，
          LLM 读着也别扭。
    [FIX] metadata 保留原值 —— 正文翻译不影响编号，编号仍是 equipment|火龙α|HELM。
    """

    _DB_NAME = "equipment"
    # 编号形如 equipment|火龙α|HELM
    _ID_FIELDS = ("series_id", "parts_type")
    FIELD_LABELS: ClassVar[dict] = {
        "series_id": "装备系列",
        "parts_type": "装备部位",
        "name": "装备名称",
        "materials": "制作材料",
    }
    VALUE_LABELS: ClassVar[dict] = {
        "parts_type": {
            "HELM": "头盔",
            "BODY": "胸甲",
            "ARM": "腕甲",
            "WAIST": "腰甲",
            "LEG": "护腿",
        },
    }


class WeaponCraftingRAG(BaseRAG):
    """
    [WHY] 「这把武器要打什么怪」不去查怪物库，靠武器表自带的 Theme 推（如火龙衍生 → 火龙）；
          少了这段正文，那道题就得多查一轮库，还什么都多不出来。
    """

    _DB_NAME = "weapon"
    # 编号形如 weapon|大剑|火龙大剑
    _ID_FIELDS = ("WeaponType", "Name")
    FIELD_LABELS: ClassVar[dict] = {
        "Name": "武器名称",
        "WeaponType": "武器类型",
        "materials": "制作材料",
        "Theme": "武器主题",
    }


class MaterialInfoRAG(BaseRAG):
    """读素材表；素材名唯一，正文只有名字 / 说明 / 稀有度，没有长叙述，单字段编号就够。"""

    _DB_NAME = "material"
    # 编号形如 material|火龙的坚壳
    _ID_FIELDS = ("RawName",)
    FIELD_LABELS: ClassVar[dict] = {
        "RawName": "素材名称",
        "RawExplain": "素材描述",
        "Rare": "稀有度",
    }


class MonsterInfoRAG(BaseRAG):
    """
    [WHY] 怪物是四个库里唯一有长叙述的，而且同一只怪的「物种 / 描述 / 生态 / 打法」回答的是
          不同问题；全拼成一坨之后，「怎么打」和「有什么生态」会匹配到同一段文字，切不开。
    """

    _DB_NAME = "monster"
    # 编号形如 monster|火龙|狩猎提示
    _ID_FIELDS = ("EnemyName", "_section")

    def _load_data(self) -> None:
        """读 monster.md 并按 # / ## 切节；每节补上「怪物名称·小节」抬头，正文没提自己是谁也能被找到。"""
        logger.info("加载怪物数据", extra={"path": self._data_path})
        # 整个 md 一次读进来，按 # / ## 切成「一只怪一小节」
        text = Path(self._data_path).read_text(encoding="utf-8")
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                # 一级标题是怪名、二级标题是小节名 —— 这两个也是编号的组成部分
                ("#", "EnemyName"),
                ("##", "_section"),
            ]
        )
        documents = splitter.split_text(text)

        # 每节补一句抬头：怪物正文常常根本不提自己叫什么，不补就搜不到
        for doc in documents:
            head = f"怪物名称：{doc.metadata['EnemyName']}，小节：{doc.metadata['_section']}。"
            doc.page_content = head + doc.page_content

        self._docs = documents
        logger.info("怪物数据加载完成", extra={"count": len(documents)})

    def retrieve(self, question: str) -> list:
        """
        [WHY] 向量分不开「火龙 / 雌火龙 / 护火龙」——名字在语义上太近，而怪物正文又常常
              根本不提自己叫什么，靠相似度定不了身份。
        """
        # 先按向量 + BM25 粗排 —— 这一步分不出名字相近的怪
        docs = self._retriever.invoke(question)

        # 再从问题里认出怪名，只留这只怪的节
        names = self._names_in_question(question)
        if names:
            kept = []
            for doc in docs:
                if doc.metadata.get("EnemyName") in names:
                    kept.append(doc)
            # 过滤到一条不剩就别过滤了：拍空不如把原始结果给回去
            if kept:
                docs = kept

        return self._top_n(docs)

    def _names_in_question(self, question: str) -> list:
        """
        [WHY] 「护火龙」三个字里含着「火龙」，直接做子串匹配会把一只怪认成两只，又串回去。
        [REF] 正向最大匹配（longest-match-first）的简化版
        """
        # rest 是「抠掉已匹配名字之后，剩下的那截问题」
        found = []
        rest = question
        # 从长到短试，长的先匹配上（_monster_names 已经按长度倒序排好）
        for name in self._monster_names():
            if name in rest:
                found.append(name)
                # 命中就抠掉，免得一只怪被认成两只
                rest = rest.replace(name, "")
        return found

    def _monster_names(self) -> list:
        """
        [WHY] 手写一份怪名词典的话，以后加怪必然忘记同步，而它坏了是静默的。
        [FIX] 实测 55 个名字无重名 —— 以后加怪要重验这一点。
        """
        # 从文档里现收集去重，不手写词典
        names = []
        for doc in self._docs:
            name = doc.metadata.get("EnemyName")
            if name and name not in names:
                names.append(name)
        # 长的排前面 —— 「最长优先」就落在这行，_names_in_question 直接吃这个顺序
        names.sort(key=len, reverse=True)
        return names
