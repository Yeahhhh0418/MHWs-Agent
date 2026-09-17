"""把待确认清单里实机核到的材料名回填进防具 jsonl。

巨戟龙β 与 α 材料完全相同（两者各槽位数量逐项一致），按 parts_type 从 α 照搬。
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TODO = ROOT / "Equipment_Material_待确认.jsonl"
DST = ROOT / "data" / "Equipment_Crafting_Table.jsonl"

_MISSING_RE = re.compile(r"^\s*x(\d+)$")
_COUNT_RE = re.compile(r"\s*x(\d+)$")
# 清单里留 ? 的条目，按 parts_type 从同系列源系列照搬
_COPY_FROM = {"巨戟龙β": "巨戟龙α"}


def _counts(materials: list) -> list:
    out = []
    for m in materials:
        mt = _COUNT_RE.search(str(m).strip())
        out.append(mt.group(1) if mt else None)
    return out


def main() -> None:
    todo = [json.loads(line) for line in TODO.open(encoding="utf-8") if line.strip()]

    # 通用占位（" x1"）补成 "? x1"，并补齐需要照搬的条目
    resolved = {}
    by_series_part = {}
    for item in todo:
        mats = [
            _MISSING_RE.sub(r"? x\1", str(m).strip()) if _MISSING_RE.match(str(m).strip()) else m
            for m in item["正确材料"]
        ]
        by_series_part[(item["系列"], item["parts_type"])] = mats
        resolved[item["装备名"]] = mats

    copied = []
    for item in todo:
        src_series = _COPY_FROM.get(item["系列"])
        if not src_series or not any("?" in str(m) for m in resolved[item["装备名"]]):
            continue
        src = by_series_part[(src_series, item["parts_type"])]
        assert not any("?" in str(m) for m in src), f"{src_series} 自身还有 ?"
        assert _counts(src) == _counts(resolved[item["装备名"]]), (
            f"{item['装备名']} 与 {src_series} 的数量对不上，不能照搬"
        )
        resolved[item["装备名"]] = list(src)
        copied.append(item["装备名"])

    rows = [json.loads(line) for line in DST.open(encoding="utf-8") if line.strip()]
    changed, skipped = [], []
    for row in rows:
        new_mats = resolved.get(row["name"])
        if new_mats is None:
            continue
        if any("?" in str(m) for m in new_mats):
            skipped.append(row["name"])
            continue
        assert _counts(new_mats) == _counts(row["materials"]), f"{row['name']} 数量被改动"
        row["materials"] = new_mats
        changed.append(row["name"])

    if skipped:
        raise SystemExit(f"以下条目仍含 ?，未回填：{skipped}")
    if len(changed) != len(todo):
        missing = {t["装备名"] for t in todo} - set(changed)
        raise SystemExit(f"有清单条目没匹配到 jsonl：{missing}")

    with DST.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"回填 {len(changed)} 条（其中 {len(copied)} 条由 α 照搬：{copied}）")


if __name__ == "__main__":
    main()
