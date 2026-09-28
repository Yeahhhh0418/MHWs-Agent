"""
四个库（装备 / 武器 / 怪物 / 素材）的 RAG 实现。
共同流程都在 rag.base.BaseRAG 里，这里只解决「各自的数据怎么读成 Document」。
"""
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from rag.base import BaseRAG
import json
from mhwi_agent.logger import get_logger
from prompts.prompt import EQUIPMENTRAG_PROMPT, MATERIALRAG_PROMPT, MONSTERRAG_PROMPT, WEAPONRAG_PROMPT

logger = get_logger("rag.main")

def load_jsonl(file_path: str) -> list:
    """读取 JSON Lines 文件，返回 list[dict]；空行跳过。"""
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))
    return data_list


class EquipmentCraftingRAG(BaseRAG):
    """
    [WHY] 装备表是英文 key，parts_type 还存着 HELM/BODY 这类枚举，直接拼进正文既会被玩家搜到，
          LLM 读着也别扭。
    [FIX] 正文用 FIELD_LABELS 的中文标签拼（装备名称：xxx），枚举再过 PARTS_TYPE_LABELS；
          metadata 保留原值，编号仍是 equipment|火龙α|HELM。
    [REF] 历史遗留-待确认
    """
    _DB_NAME = "equipment"
    _ID_FIELDS = ("series_id","parts_type")
    RAG_PROMPT_TEMPLATE = EQUIPMENTRAG_PROMPT
    FIELD_LABELS = {
        "series_id": "装备系列",
        "parts_type": "装备部位",
        "name": "装备名称",
        "materials": "制作材料",
    }
    PARTS_TYPE_LABELS = {
        "HELM": "头盔",
        "BODY": "胸甲",
        "ARM": "腕甲",
        "WAIST": "腰甲",
        "LEG": "护腿",
    }
    def _load_data(self):
        """读装备表，按「标签：值」拼成正文；字段名和枚举都先翻成中文（见类注释）。"""
        logger.info("加载装备数据", extra={"path": self._data_path})
        documents = []

        data_list=load_jsonl(self._data_path)
        for data in data_list:
            parts = []
            for k, v in data.items():
                if k == "parts_type" and v in self.PARTS_TYPE_LABELS:
                    v = self.PARTS_TYPE_LABELS[v]

                label = self.FIELD_LABELS.get(k, k)

                if isinstance(v, list):
                    value_str = "、".join(str(item) for item in v)
                else:
                    value_str = str(v)

                parts.append(f"{label}：{value_str}")

            content = "，".join(parts)

            documents.append(Document(
                page_content=content,
                metadata=data,
            ))

        self._docs = documents
        logger.info("装备数据加载完成", extra={"count": len(documents)})

class WeaponCraftingRAG(BaseRAG):
    """
    [WHY] 「这把武器要打什么怪」不去查怪物库，靠武器表自带的 Theme 推（如火龙衍生 → 火龙）；
          少了这段正文，那道题就得多查一轮库，还什么都多不出来。
    [FIX] FIELD_LABELS 把 Theme 拼成「武器主题：火龙衍生」进正文，武器这一条自己就能答。
    [REF] advise/武器主题推断方案.md
    """
    _DB_NAME = "weapon"
    _ID_FIELDS = ("WeaponType","Name")
    RAG_PROMPT_TEMPLATE = WEAPONRAG_PROMPT
    FIELD_LABELS = {
        "Name": "武器名称",
        "WeaponType": "武器类型",
        "materials": "制作材料",
        "Theme": "武器主题",
    }
    def _load_data(self):
        """读武器表，按「标签：值」拼成正文；Theme 必须进正文，路由靠它推「打什么怪」。"""
        logger.info("加载武器数据", extra={"path": self._data_path})
        documents = []

        data_list=load_jsonl(self._data_path)
        for data in data_list:
            parts = []
            for k, v in data.items():
                label = self.FIELD_LABELS.get(k, k)
                if isinstance(v, list):
                    value_str = "、".join(str(item) for item in v)
                else:
                    value_str = str(v)
                parts.append(f"{label}：{value_str}")
            content = "，".join(parts)

            documents.append(Document(
                page_content=content,
                metadata=data,
            ))

        self._docs = documents
        logger.info("武器数据加载完成", extra={"count": len(documents)})



class MonsterInfoRAG(BaseRAG):
    """
    [WHY] 怪物是四个库里唯一有长叙述的，而且同一只怪的「物种 / 描述 / 生态 / 打法」回答的是
          不同问题；全拼成一坨之后，「怎么打」和「有什么生态」会匹配到同一段文字，切不开。
    [FIX] 数据改存 monster.md，按 # 分怪、## 分节，一节只讲一件事，编号取「怪名|小节」两段。
    [REF] 怪物库改造指南.md
    """
    _DB_NAME = "monster"
    _ID_FIELDS = ("EnemyName", "_section")
    RAG_PROMPT_TEMPLATE = MONSTERRAG_PROMPT

    def _load_data(self):
        """读 monster.md 并按 # / ## 切节；每节补上「怪物名称·小节」抬头，正文没提自己是谁也能被找到。"""
        logger.info("加载怪物数据", extra={"path": self._data_path})
        text = Path(self._data_path).read_text(encoding="utf-8")
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "EnemyName"),
                ("##", "_section"),
            ]
        )
        documents = splitter.split_text(text)

        for doc in documents:
            head = f"怪物名称：{doc.metadata['EnemyName']}，小节：{doc.metadata['_section']}。"
            doc.page_content = head + doc.page_content

        self._docs = documents
        logger.info("怪物数据加载完成", extra={"count": len(documents)})

    def retrieve(self, question: str) -> list:
        """
        [WHY] 向量分不开「火龙 / 雌火龙 / 护火龙」——名字在语义上太近，而怪物正文又常常
              根本不提自己叫什么，靠相似度定不了身份。
        [FIX] 先用名字表认出问题里是哪只怪（长名优先），命中就按 EnemyName 过滤候选；
              一个名字都没认出来（如「天空王者」这类别名）就退回原来的向量检索。
        [REF] 2026-09-28 排查记录；可对照 scripts/probe_monster_retrieval.py 复现
        """
        docs = self._retriever.invoke(question)

        names = self._names_in_question(question)
        if names:
            kept = []
            for doc in docs:
                if doc.metadata.get("EnemyName") in names:
                    kept.append(doc)
            if kept:
                docs = kept

        return self._top_n(docs)

    def _names_in_question(self, question: str) -> list:
        """
        [WHY] 「护火龙」三个字里含着「火龙」，直接做子串匹配会把一只怪认成两只，又串回去。
        [FIX] 名字按长度倒序去比，认到就从问题里划掉，同一只怪只认一次。
        [REF] 正向最大匹配（longest-match-first）的简化版
        """
        found = []
        rest = question
        for name in self._monster_names():
            if name in rest:
                found.append(name)
                rest = rest.replace(name, "")
        return found

    def _monster_names(self) -> list:
        """
        [WHY] 手写一份怪名词典的话，以后加怪必然忘记同步，而它坏了是静默的。
        [FIX] 名字表直接从已加载的 _docs 里现收（55 个，实测无重名），按长度倒序供上面用。
        [REF] 2026-09-28 加入
        """
        names = []
        for doc in self._docs:
            name = doc.metadata.get("EnemyName")
            if name and name not in names:
                names.append(name)
        names.sort(key=len, reverse=True)
        return names


class MaterialInfoRAG(BaseRAG):
    """读素材表；素材名唯一，正文只有名字 / 说明 / 稀有度，没有长叙述，单字段编号就够。"""
    _DB_NAME = "material"
    _ID_FIELDS = ("RawName",)
    RAG_PROMPT_TEMPLATE = MATERIALRAG_PROMPT
    FIELD_LABELS = {
        "RawName": "素材名称",
        "RawExplain": "素材描述",
        "Rare": "稀有度",
        }
    def _load_data(self):
        """读素材表，按「标签：值」拼成正文。"""
        logger.info("加载素材数据", extra={"path": self._data_path})
        documents = []

        data_list=load_jsonl(self._data_path)
        for data in data_list:
            parts = []
            for k, v in data.items():
                label = self.FIELD_LABELS.get(k, k)
                if isinstance(v, list):
                    value_str = "、".join(str(item) for item in v)
                else:
                    value_str = str(v)
                parts.append(f"{label}：{value_str}")
            content = "，".join(parts)

            documents.append(Document(
                page_content=content,
                metadata=data,
            ))

        self._docs = documents
        logger.info("素材数据加载完成", extra={"count": len(documents)})



if __name__ == "__main__":
    # [TODO-TECHDEBT] 手工调试脚本，四个库靠注释来回切换；建议挪到 scripts/ 或改成 pytest 参数化
    """
    rag = EquipmentCraftingRAG()
    rag.setup()
    docs = rag.retrieve("巨戟龙套怎么制作")
    for i, doc in enumerate(docs, 1):
        print(f"{'─'*50}")
        print(f"  [{i}] {doc.page_content}")
    print(f"{'─'*50}")
    print(f"共检索到 {len(docs)} 条结果")
    print(rag("巨戟龙套怎么制作"))
    """

    """
    rag2 = WeaponCraftingRAG()
    rag2.setup()
    docs = rag2.retrieve("刺花蜘蛛太刀怎么做")
    for i, doc in enumerate(docs, 1):
        print(f"{'─'*50}")
        print(f"  [{i}] {doc.page_content}")
    print(f"{'─'*50}")
    print(f"共检索到 {len(docs)} 条结果")
    print(rag2("刺花蜘蛛太刀怎么做"))
    """


    #"""
    rag3 = MonsterInfoRAG()
    rag3.setup()
    docs = rag3.retrieve("火龙怎么打")
    for i, doc in enumerate(docs, 1):
        print(f"{'─'*50}")
        print(f"  [{i}] {doc}")
    print(f"{'─'*50}")
    print(f"共检索到 {len(docs)} 条结果")
    print(rag3("火龙怎么打"))
    #"""

    """
    rag4 = MaterialInfoRAG()
    rag4.setup()
    docs = rag4.retrieve("锁刃龙素材")
    for i, doc in enumerate(docs, 1):
        print(f"{'─'*50}")
        print(f"  [{i}] {doc.page_content}")
    print(f"{'─'*50}")
    print(f"共检索到 {len(docs)} 条结果")
    print(rag4("锁刃龙的素材"))
    """
