"""
agent/rag/main_rag.py —— 装备制作 RAG（混合检索）
"""
from langchain_core.documents import Document
from rag.base import BaseRAG
from mhwi_agent.logger import get_logger
from prompts.prompt import EQUIPMENTRAG_PROMPT, MATERIALRAG_PROMPT, MONSTERRAG_PROMPT, WEAPONRAG_PROMPT

logger = get_logger("rag.main")

class EquipmentCraftingRAG(BaseRAG):
    _DB_NAME = "equipment"
    RAG_PROMPT_TEMPLATE = EQUIPMENTRAG_PROMPT
    def _load_data(self):
        """加载装备 JSON，转为 Document 列表。"""
        logger.info("加载装备数据", extra={"path": self._data_path})
        documents = []

        # 字段含义映射：key → 中文描述
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
        data_list=BaseRAG.load_jsonl(self._data_path)  # 读取 JSON Lines 文件，返回解析后的数据列表
        for data in data_list:
            # 所有字段拼成一段描述
            parts = []
            for k, v in data.items():
                # 1. 先处理枚举值映射（保持你原有的业务逻辑）
                if k == "parts_type" and v in PARTS_TYPE_LABELS:
                    v = PARTS_TYPE_LABELS[v]
                
                label = FIELD_LABELS.get(k, k)
                
                # 2. 列表字段用顿号连接，非列表直接转字符串
                if isinstance(v, list):
                    value_str = "、".join(str(item) for item in v)
                else:
                    value_str = str(v)
                    
                parts.append(f"{label}：{value_str}")
            
            content = "，".join(parts)

            #将处理完的content,sanitized封装成 Document 对象
            documents.append(Document(
                page_content=content,
                metadata=data,
            ))  

        self._docs = documents
        logger.info("装备数据加载完成", extra={"count": len(documents)})

class WeaponCraftingRAG(BaseRAG):
    _DB_NAME = "weapon"
    RAG_PROMPT_TEMPLATE = WEAPONRAG_PROMPT
    def _load_data(self):
        """加载武器 JSON，转为 Document 列表。"""
        logger.info("加载武器数据", extra={"path": self._data_path})
        documents = []
        FIELD_LABELS={
            "Name": "武器名称",
            "WeaponType": "武器类型",
            "materials": "制作材料",
        }
        data_list=BaseRAG.load_jsonl(self._data_path)
        for data in data_list:
            parts = []
            for k, v in data.items():
                label = FIELD_LABELS.get(k, k)
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
    _DB_NAME = "monster"
    RAG_PROMPT_TEMPLATE = MONSTERRAG_PROMPT

    # 常规字段的中文标签
    FIELD_LABELS = {
        "EnemyName": "怪物名称",
        "Species": "种类",
        "MonsterSize": "体型",
        "EnemyExp": "怪物描述",
        "EnemyFeatures": "生态特征",
        "EnemyTips": "狩猎提示",
        "EnemyLegendaryName": "历战个体",
    }

    SIZE_LABELS = {
        "大型": "大型怪物（正式狩猎目标）",
        "小型": "小型怪物（成群出没的杂兵，非正式狩猎目标）",
    }

    RARE_FIELD_GLOSSARY = {
        "EnemyLegendaryKingName": ("历战王名", "同一只怪物的最强强化版，比历战个体更强，本作只有少数怪物有"),
        "EnemyFrenzyName":        ("狂龙化名", "被狂龙病毒强化后的形态名，攻击更狂暴，不是新怪物"),
        "EnemyExtraName":         ("特殊个体别名", "非标准形态的称呼，如首领个体、克隆体、家畜种"),
        "EnemyBossExp":           ("首领／幼体说明", "对上面这类特殊形态的补充描述，不是另一只怪物的简介"),
    }

    def _load_data(self):
        """加载怪物 JSON，转为 Document 列表。"""
        logger.info("加载怪物数据", extra={"path": self._data_path})
        documents = []
        data_list = BaseRAG.load_jsonl(self._data_path)
        for data in data_list:
            row = dict(data)
            size_text = self.SIZE_LABELS.get(str(row.get("MonsterSize", "")).strip())
            if size_text:
                row["MonsterSize"] = size_text

            # 所有字段拼成一段描述
            parts = []
            notes = []   # 只收集本条记录真正出现的稀有字段
            for k, v in row.items():
                if not str(v).strip():
                    # 值为空就不渲染这个字段，免得留下"生态特征："这种悬空标签
                    continue

                if k in self.RARE_FIELD_GLOSSARY:
                    label, gloss = self.RARE_FIELD_GLOSSARY[k]
                    notes.append(f"{label} = {gloss}")
                else:
                    label = self.FIELD_LABELS.get(k, k)
                parts.append(f"{label}：{v}")

            content = "，".join(parts)
            if notes:
                # 说明文字也会进 BM25／向量索引，所以只给带字段的记录加，
                # 不让其余记录平白多出这些词
                content += "\n【本条额外字段说明】" + "；".join(notes)

            # 将处理完的 content、原始字段封装成 Document 对象
            documents.append(Document(
                page_content=content,
                metadata=data,
            ))

        self._docs = documents
        logger.info("怪物数据加载完成", extra={"count": len(documents)})

class MaterialInfoRAG(BaseRAG):
    _DB_NAME = "material"
    RAG_PROMPT_TEMPLATE = MATERIALRAG_PROMPT


if __name__ == "__main__":
    # 测试加载装备数据
    """
    rag = EquipmentCraftingRAG()
    rag.setup()
    docs = rag.retrieve("豪鬼套怎么制作")
    for i, doc in enumerate(docs, 1):
        print(f"{'─'*50}")
        print(f"  [{i}] {doc.page_content}")
    print(f"{'─'*50}")
    print(f"共检索到 {len(docs)} 条结果")
    print(rag("豪鬼套怎么制作"))
    """

    """
    rag2 = WeaponCraftingRAG()
    rag2.setup()
    docs = rag2.retrieve("刺花蜘蛛大剑怎么做")
    for i, doc in enumerate(docs, 1):
        print(f"{'─'*50}")
        print(f"  [{i}] {doc.page_content}")
    print(f"{'─'*50}")
    print(f"共检索到 {len(docs)} 条结果")
    print(rag2("刺花蜘蛛大剑怎么做"))
    """


    rag3 = MonsterInfoRAG()
    rag3.setup()
    docs = rag3.retrieve("护锁刃龙")
    for i, doc in enumerate(docs, 1):
        print(f"{'─'*50}")
        print(f"  [{i}] {doc.page_content}")
    print(f"{'─'*50}")
    print(f"共检索到 {len(docs)} 条结果")
    print(rag3("锁刃龙"))

