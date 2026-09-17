"""找出防具里材料名缺失（只剩数量）的条目，导出待确认清单。

缺失项在源表「装备合成表 1.040.xlsx」里就没有名字，只能实机核对后回填。
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "Equipment_Crafting_Table.jsonl"
DST = ROOT / "Equipment_Material_待确认.jsonl"

_MISSING_RE = re.compile(r"^\s*x(\d+)$")
_PARTS_ORDER = ["HELM", "BODY", "ARM", "WAIST", "LEG"]


def main() -> None:
    rows = [json.loads(line) for line in SRC.open(encoding="utf-8") if line.strip()]

    todo = []
    for row in rows:
        materials = row.get("materials") or []
        if not any(_MISSING_RE.match(str(m).strip()) for m in materials):
            continue
        placeholders = [
            _MISSING_RE.sub(r"? x\1", str(m).strip()) if _MISSING_RE.match(str(m).strip()) else m
            for m in materials
        ]
        todo.append({
            "系列": row["series_id"],
            "parts_type": row["parts_type"],
            "装备名": row["name"],
            "正确材料": placeholders,
        })

    series_order = {s: i for i, s in enumerate(dict.fromkeys(r["series_id"] for r in rows))}
    todo.sort(key=lambda x: (series_order[x["系列"]], _PARTS_ORDER.index(x["parts_type"])))

    with DST.open("w", encoding="utf-8") as f:
        for item in todo:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"写入 {DST}：{len(todo)} 条，涉及 {len({t['系列'] for t in todo})} 个系列")


if __name__ == "__main__":
    main()
