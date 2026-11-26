#!/usr/bin/env python
"""Merge supervised pair files while preserving source/unit provenance."""
import json
import os
import sys
from collections import Counter


def main(out_path, *specs):
    rows = []
    for spec in specs:
        source, path = spec.split("=", 1)
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            r["source_dataset"] = source
            r.setdefault("source_unit", "page" if source == "mmlongbench_doc" else "paragraph")
            rows.append(r)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"pairs": len(rows), "questions": len({r["qid"] for r in rows}),
             "by_source": dict(Counter(r["source_dataset"] for r in rows)),
             "by_label": dict(Counter(str(r["label"]) for r in rows)),
             "note": "mixed training only; evaluation remains source-specific"}
    json.dump(stats, open(out_path + ".stats.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
