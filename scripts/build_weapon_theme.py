"""从「武器制造派生树 1.040.xlsx」提取衍生主题，写入武器 jsonl 的新字段 Theme。

派生树每行 A 列是主题（如「护火龙衍生」），奇数列为树中武器名（带 (火)/(冰) 属性后缀）。
1070 条能一一对上；对不上的 28 条见 _OVERRIDE。
"""

import json
import re
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "game_data" / "武器制造派生树 1.040.xlsx"
SRC = ROOT / "data" / "Weapon_Crafting_Table.jsonl"
DST = ROOT / "Weapon_Crafting_Table_with_theme.jsonl"

# 派生树里没有这 28 条。前两批是树的 sheet 漏画了行（另几个类型有同名主题且材料一致），
# 后两批经实机核对补上。
_OVERRIDE = {
    **{n: "海龙衍生" for n in (
        "海雷涡流锤Ⅰ", "海雷涡流猛锤", "海龙角笛Ⅰ", "雷笛雷电角",
        "十字长枪Ⅰ", "高级十字长枪", "雷光战斧Ⅰ", "高压雷光战斧",
        "海龙暴风雨Ⅰ", "雷盾斧胜者漩涡", "雷光弓Ⅰ", "高压雷光弓",
        "海龙闪击Ⅰ", "雷炮海龙电击", "雷炮雷霆交叉Ⅰ", "雷迅炮雷霆交叉",
    )},
    # 欧米茄之剑（单手剑）在树里标了「欧米茄衍生」，这几条材料与它逐项一致。
    **{n: "欧米茄衍生" for n in (
        "暗影使者", "暗影使者·真", "欧米茄之杖", "欧米茄之杖·真",
    )},
    **{n: "艾露猫衍生" for n in ("猫？拳", "喵喵棒")},
    **{n: "工房武器衍生" for n in (
        "美味双剑", "全熟双剑", "钢铁十字镐", "钢铁十字镐【采掘狂】",
        "玉米炮", "玉米大炮",
    )},
}

_TREE_CHARS = set("│─├└┌┐┘┼ ")

# 源表里这两条的材料名缺失，只剩数量（如 " x2"），实机查到是「捣蛋猫的通缉令」。
_MISSING_MATERIAL = "捣蛋猫的通缉令"
_MISSING_MATERIAL_RE = re.compile(r"^\s*x(\d+)$")


def _cell(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if all(ch in _TREE_CHARS for ch in text) else text


def _base(name: str) -> str:
    """去掉树里的属性后缀，如「白炎翼剑Ⅰ (火)」→「白炎翼剑Ⅰ」。"""
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()


def _fix_materials(materials: list) -> list:
    return [_MISSING_MATERIAL_RE.sub(rf"{_MISSING_MATERIAL} x\1", m) for m in materials]


def load_themes() -> dict:
    wb = openpyxl.load_workbook(XLSX, data_only=True, read_only=True)
    themes: dict[str, str] = {}
    for ws in wb.worksheets:
        for row in ws.iter_rows(min_row=2, values_only=True):
            theme = _cell(row[0])
            if not theme or "AUTO-GE" in theme:
                continue
            for i in range(1, len(row), 2):
                name = _cell(row[i])
                if name:
                    themes[_base(name)] = theme
    wb.close()
    return themes


def main() -> None:
    themes = load_themes()
    rows = [json.loads(line) for line in SRC.open(encoding="utf-8") if line.strip()]

    missing = []
    out = []
    for row in rows:
        name = row["Name"]
        theme = themes.get(name) or _OVERRIDE.get(name)
        if theme is None:
            missing.append(name)
        new = {k: row[k] for k in ("Name", "WeaponType", "materials") if k in row}
        new.update({k: v for k, v in row.items() if k not in new})
        if "materials" in new:
            new["materials"] = _fix_materials(new["materials"])
        if theme:
            new["Theme"] = theme
        out.append(new)

    with DST.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"写入 {DST}：{len(out)} 条，其中有 Theme {sum('Theme' in r for r in out)} 条")
    print(f"无 Theme：{len(missing)} 条 {missing}")
    print(f"主题种类：{len(set(r['Theme'] for r in out if 'Theme' in r))}")


if __name__ == "__main__":
    main()
