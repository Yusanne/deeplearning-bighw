#!/usr/bin/env python
"""Week1 S1: select 20 feasibility questions from MMLongBench-Doc samples.json.

Usage: python -I s1_select.py <samples.json> <out_selected.json>
"""
import ast
import json
import random
import re
import sys
from collections import Counter


def parse_maybe_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            x = ast.literal_eval(v)
            return x if isinstance(x, list) else [x]
        except Exception:
            return [v]
    return [v]


def norm(s):
    return re.sub(r"[%$€£'\"\s(),.]+", "", str(s)).lower()


def answerable(item):
    fmt = str(item.get("answer_format", "")).strip().lower()
    if fmt.startswith("null") or fmt in ("none", ""):
        return False
    a = norm(item.get("answer", ""))
    if a in ("", "notanswerable", "unanswerable", "noanswer", "na"):
        return False
    return True


def main(samples_path, out_path):
    data = json.load(open(samples_path, encoding="utf-8"))
    items = data["samples"] if isinstance(data, dict) and "samples" in data else data
    stats = {"total": len(items)}
    fmt_counter = Counter(str(it.get("answer_format")) for it in items)
    src_counter = Counter()
    unanswerable = 0
    pure_single_pool = []
    pure_multi_pool = []
    for it in items:
        ev_srcs = [str(s) for s in parse_maybe_list(it.get("evidence_sources", "[]"))]
        ev_pages = parse_maybe_list(it.get("evidence_pages", "[]"))
        key = "+".join(sorted(set(ev_srcs))) if ev_srcs else "unknown"
        src_counter[key] += 1
        if not answerable(it):
            unanswerable += 1
            continue
        if not ev_srcs or not all("Pure-text" in s for s in ev_srcs):
            continue
        ans = str(it.get("answer", "")).strip()
        if not (1 <= len(ans) <= 30):
            continue
        rec = {
            "doc_id": it["doc_id"],
            "doc_type": it.get("doc_type", ""),
            "question": it["question"],
            "answer": ans,
            "answer_format": str(it.get("answer_format")),
            "evidence_pages": [int(p) for p in ev_pages if str(p).strip().lstrip("-").isdigit()],
            "evidence_sources": ev_srcs,
        }
        if len(ev_pages) == 1:
            pure_single_pool.append(rec)
        else:
            pure_multi_pool.append(rec)

    stats.update(
        {
            "answer_format_counts": dict(fmt_counter),
            "unanswerable_est": unanswerable,
            "unanswerable_rate": round(unanswerable / max(stats["total"], 1), 4),
            "pure_text_single_evidence_pool": len(pure_single_pool),
            "pure_text_multi_evidence_pool": len(pure_multi_pool),
        }
    )

    seed = 2024201433
    rng = random.Random(seed)
    rng.shuffle(pure_single_pool)
    selected, per_doc = [], Counter()
    for rec in pure_single_pool:
        if len(selected) >= 20:
            break
        if per_doc[rec["doc_id"]] >= 3:
            continue
        selected.append(rec)
        per_doc[rec["doc_id"]] += 1
    for i, rec in enumerate(selected, 1):
        rec["qid"] = f"cfw1_{i:03d}"

    stats["selected"] = len(selected)
    stats["selected_docs"] = len(per_doc)
    out = {"seed": seed, "stats": stats, "questions": selected,
           "doc_ids": sorted(per_doc)}
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("selected:", len(selected), "docs:", len(per_doc))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
