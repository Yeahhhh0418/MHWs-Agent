"""
纯数据校验：不加载模型、不联网，秒出结果。

测的是「编号」这套东西有没有坏 —— 它一坏就是静默的：召回率悄悄变成 0，却不报错。
"""

import json
from pathlib import Path

from app.rag import mainrag
from app.rag.base import BaseRAG

ROOT = Path(__file__).resolve().parent.parent

# 条数写死是故意的：数据变了就得有人主动来改这里，而不是悄悄通过
LIBRARIES = [
    (mainrag.EquipmentCraftingRAG, 693),
    (mainrag.WeaponCraftingRAG, 1098),
    (mainrag.MaterialInfoRAG, 497),
    (mainrag.MonsterInfoRAG, 202),
]


def load_rag(cls) -> BaseRAG:
    """只跑数据加载和编号，不碰向量库和模型 —— 这几条测试能 4 秒跑完就靠它。"""
    rag = cls()
    rag._load_data()
    rag._assign_doc_ids()
    return rag


def load_cases() -> list:
    """读题库，返回 list[dict]。"""
    cases = []
    with open(ROOT / "src" / "app" / "eval" / "cases.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def test_library_counts() -> None:
    """防：数据文件被改坏，或 _load_data 出问题，入库条数悄悄变了却没人发现。"""
    for cls, expected in LIBRARIES:
        rag = cls()
        rag._load_data()
        assert len(rag._docs) == expected, (
            f"{cls.__name__} 入库 {len(rag._docs)} 条，预期 {expected} 条"
        )


def test_every_doc_id_is_unique() -> None:
    """防：编号重复 —— 检索去重靠 _doc_id，重复了会认错条目。"""
    for cls, _ in LIBRARIES:
        rag = load_rag(cls)

        seen = {}
        for doc in rag._docs:
            one_id = doc.metadata["_doc_id"]
            seen[one_id] = seen.get(one_id, 0) + 1

        duplicated = []
        for one_id, n in seen.items():
            if n > 1:
                duplicated.append(one_id)

        assert duplicated == [], f"{cls.__name__} 有重复编号：{duplicated[:5]}"


def test_case_doc_ids_exist() -> None:
    """防：题库和数据脱节 —— 改了标题或 _ID_FIELDS 之后，召回率会静默变 0。"""
    all_ids = set()
    for cls, _ in LIBRARIES:
        rag = load_rag(cls)
        for doc in rag._docs:
            all_ids.add(doc.metadata["_doc_id"])

    missing = []
    for case in load_cases():
        for one_id in case["required_doc_ids"]:
            if one_id not in all_ids:
                missing.append((case["question"], one_id))

    assert missing == [], f"题库里有 {len(missing)} 个编号对不上：{missing}"


def test_case_doc_prefix_matches_tags() -> None:
    """防：题库标注写错库，比如把 monster 的题标成了 weapon。"""
    for case in load_cases():
        for one_id in case["required_doc_ids"]:
            prefix = one_id.split("|")[0]
            assert prefix in case["tags"], (
                f"「{case['question']}」的编号 {one_id} 属于 {prefix}，但 tags 里没有它"
            )
