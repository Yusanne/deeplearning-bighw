#!/usr/bin/env python3
"""Create a fair two-person CiteFaith-20 review assignment.

The formal audit contains 20 unique candidates:
7 Pan-only + 8 Zhang-only + 5 double-reviewed = 20 unique items.
This keeps the two review workloads within one action while preserving
overlap for agreement analysis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "project/results/raw/audit_candidates.json"
OUTPUT_DIR = ROOT / "project/results/raw"
FULL_OUTPUT = OUTPUT_DIR / "review_assignments.csv"
PAN_OUTPUT = OUTPUT_DIR / "review_task_pan.csv"
ZHANG_OUTPUT = OUTPUT_DIR / "review_task_zhang.csv"


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(clean(item) for item in value)
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def load_candidates() -> list[dict[str, Any]]:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    candidates = [item for item in payload["audit_set"] if item.get("status") == "candidate_valid"]
    if len(candidates) < 20:
        raise RuntimeError(f"需要至少20个有效候选，实际只有{len(candidates)}个")
    return candidates[:20]


def assignment_for(index: int) -> tuple[str, str, str]:
    if index < 5:
        return "共同复核", "潘雨萱;张轩晨", "张轩晨"
    if index < 12:
        return "潘雨萱独立主审", "潘雨萱", ""
    return "张轩晨独立主审", "张轩晨", ""


def row_for(index: int, item: dict[str, Any]) -> dict[str, str]:
    mode, reviewers, secondary = assignment_for(index)
    primary = "共同" if index < 5 else ("潘雨萱" if index < 12 else "张轩晨")
    return {
        "assignment_id": f"cf20_{index + 1:03d}",
        "pilot_qid": clean(item.get("pilot_qid")),
        "doc_id": clean(item.get("doc_id")),
        "doc_type": clean(item.get("doc_type")),
        "question": clean(item.get("question")),
        "gold_answer": clean(item.get("answer")),
        "gold_pages": clean(item.get("evidence_pages")),
        "swap_page": clean(item.get("swap_page")),
        "swap_rule": clean(item.get("swap_rule")),
        "swap_replacement": clean(item.get("swap_replacement")),
        "benign_pages": clean(item.get("benign_pages")),
        "primary_reviewer": primary,
        "reviewers": reviewers,
        "secondary_reviewer": secondary,
        "review_mode": mode,
        "pdf_path": f"project/data/documents/{clean(item.get('doc_id'))}",
        "page_text_path": f"project/data/pages/{clean(item.get('doc_id'))}.json",
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    candidates = load_candidates()
    rows = [row_for(index, item) for index, item in enumerate(candidates)]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(FULL_OUTPUT, rows)
    write_csv(PAN_OUTPUT, [row for row in rows if "潘雨萱" in row["reviewers"]])
    write_csv(ZHANG_OUTPUT, [row for row in rows if "张轩晨" in row["reviewers"]])
    print(f"wrote {FULL_OUTPUT}: {len(rows)} unique candidates")
    print(f"wrote {PAN_OUTPUT}: {sum('潘雨萱' in row['reviewers'] for row in rows)} review actions")
    print(f"wrote {ZHANG_OUTPUT}: {sum('张轩晨' in row['reviewers'] for row in rows)} review actions")
    print("overlap: 5; pan-only: 7; zhang-only: 8")


if __name__ == "__main__":
    main()
