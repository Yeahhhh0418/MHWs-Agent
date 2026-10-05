"""把 data/Monster_info_Table.jsonl 合并转成 data/monster.md。

⚠️ 这不是一次性脚本 —— 游戏更新、怪物表变了之后要**重跑**。
   删了它，怪物表变了就没法同步了。

⚠️ monster.md 里的 # / ## 标题就是编号的来源（monster|火龙|狩猎提示）。
   改了下面的字段映射，要同步改子类的 _ID_FIELDS 和 cases.jsonl 的 required_doc_ids，
   缺一样召回率就悄悄变 0 —— 不报错的那种坏（CLAUDE.md 硬规矩 1）。

在项目根目录执行：.venv/Scripts/python.exe scripts/build_monster_md.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "Monster_info_Table.jsonl"
DST = ROOT / "data" / "monster.md"

BASIC_FIELDS = [
    ("Species", "种类"),
    ("MonsterSize", "体型"),
    ("EnemyLegendaryName", "历战个体"),
    ("EnemyLegendaryKingName", "历战王名"),
    ("EnemyFrenzyName", "狂龙化名"),
    ("EnemyExtraName", "特殊个体别名"),
]

SECTION_FIELDS = [
    ("EnemyExp", "怪物描述"),
    ("EnemyFeatures", "生态特征"),
    ("EnemyTips", "狩猎提示"),
    ("EnemyBossExp", "首领／幼体说明"),
]

GLOSSARY = {
    "EnemyLegendaryKingName": "同一只怪物的最强强化版，比历战个体更强，本作只有少数怪物有",
    "EnemyFrenzyName": "被狂龙病毒强化后的形态名，攻击更狂暴，不是新怪物",
    "EnemyExtraName": "非标准形态的称呼，如首领个体、克隆体、家畜种",
    "EnemyBossExp": "对上面这类特殊形态的补充描述，不是另一只怪物的简介",
}

SIZE_LABELS = {
    "大型": "大型怪物（正式狩猎目标）",
    "小型": "小型怪物（成群出没的杂兵，非正式狩猎目标）",
}


def load_rows() -> list:
    rows = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def check_unique_names(rows: list) -> None:
    seen = set()
    for row in rows:
        name = row["EnemyName"]
        if name in seen:
            raise SystemExit(f"怪物名重复：{name}。同一个名字只能出现一次，改了再跑。")
        seen.add(name)


def is_empty(value: object) -> bool:
    return value is None or str(value).strip() == ""


def basic_block(row: dict) -> list:
    lines = []
    for key, label in BASIC_FIELDS:
        value = row.get(key)
        if is_empty(value):
            continue
        if key == "MonsterSize":
            text = SIZE_LABELS.get(str(value).strip(), str(value))
        else:
            text = str(value)
        gloss = GLOSSARY.get(key)
        if gloss:
            text = f"{text}（{gloss}）"
        lines.append(f"- {label}：{text}")
    return lines


def section_block(row: dict, key: str, label: str) -> list:
    value = row.get(key)
    if is_empty(value):
        return []
    text = str(value)
    gloss = GLOSSARY.get(key)
    if gloss:
        text = f"{text}（{gloss}）"
    return [f"## {label}", text, ""]


def build_md(rows: list) -> str:
    parts = []
    for row in rows:
        parts.append(f"# {row['EnemyName']}")
        parts.append("")

        basic = basic_block(row)
        if basic:
            parts.append("## 基本信息")
            parts.extend(basic)
            parts.append("")

        for key, label in SECTION_FIELDS:
            parts.extend(section_block(row, key, label))

    text = "\n".join(parts)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.rstrip() + "\n"


def main() -> None:
    rows = load_rows()
    check_unique_names(rows)
    text = build_md(rows)
    DST.write_text(text, encoding="utf-8")
    print(f"写入 {len(rows)} 只怪物 -> {DST}")


if __name__ == "__main__":
    main()
