#!/usr/bin/env python
"""S4: autofilter the audit pool (§3.5.4): redundancy check -> swap construction -> benign pages.
Selects the CiteFaith audit set (target 40) + a 10-question diverse review shortlist for the
second annotator. Verifies the ">=20 low-redundancy candidates" gate.

Usage: python -I s4_autofilter_audit.py <audit_pool.json> <pages_dir> <outdir>
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_cf as L


def main(pool_path, pages_dir, outdir):
    pool = json.load(open(pool_path, encoding="utf-8"))["questions"]
    rows = []
    for r in pool:
        pages = L.load_pages(pages_dir, r["doc_id"])
        if pages is None:
            rows.append({**r, "status": "no_pages_file"})
            continue
        corpus = L.corpus_pages(pages)
        if not corpus:
            rows.append({**r, "status": "no_text_layer"})
            continue
        gold, gold_src = L.gold_pages_by_answer(pages, r["answer"], r["evidence_pages"])
        if gold_src != "answer-string":
            rows.append({**r, "status": "gold_not_located", "gold_src": gold_src})
            continue
        g = L.norm(r["answer"])
        redundant_pages = [i for i in corpus if i not in gold and g in L.norm(pages[i])]
        bm, tf, rrf = L.retrieve(pages, r["question"])
        swap_page = gold[0]
        ok, rule, e_minus, repl, reason = L.build_swap(pages[swap_page], r["answer"])
        valid = ok and L.swap_valid_proxy(pages[swap_page], e_minus, r["answer"], repl)
        benign = L.benign_candidates(pages, r["answer"], gold, swap_page, repl, bm)
        if redundant_pages:
            status = "redundant_support"
        elif valid:
            status = "candidate_valid"
        elif ok:
            status = "candidate_weak"  # swap built but gold leaked / replacement missing
        else:
            status = "swap_invalid"
        rows.append({**r, "status": status, "gold_pages": gold, "gold_src": gold_src,
                     "redundant_pages": redundant_pages, "swap_page": swap_page,
                     "swap_rule": rule, "swap_replacement": repl,
                     "benign_pages": benign, "rrf_rank_of_gold": next(
                         (i + 1 for i, p in enumerate(rrf) if p in gold), None)})

    valid = [r for r in rows if r["status"] == "candidate_valid"]
    pilots = [r for r in valid if r.get("pilot_qid")]
    non_pilot = [r for r in valid if not r.get("pilot_qid")]
    # prefer single-evidence, diversify distractor type
    non_pilot.sort(key=lambda r: (not r.get("pref_single_evidence", False), r["swap_rule"]))
    by_rule, selected = {}, list(pilots)
    for r in pilots:
        by_rule[r["swap_rule"]] = by_rule.get(r["swap_rule"], 0) + 1
    for r in non_pilot:
        if len(selected) >= 40:
            break
        selected.append(r)
        by_rule[r["swap_rule"]] = by_rule.get(r["swap_rule"], 0) + 1

    # review shortlist: 10 diverse, include 4 pilot (already human-reviewed once) + 6 new
    short = [r for r in selected if r.get("pilot_qid")][:4]
    seen = {r.get("pilot_qid") for r in short}
    for rule in sorted(by_rule):
        for r in selected:
            if len(short) >= 10:
                break
            if r in short or r["swap_rule"] != rule:
                continue
            short.append(r)
        if len(short) >= 10:
            break
    for r in selected:
        if len(short) >= 10:
            break
        if r not in short:
            short.append(r)

    os.makedirs(outdir, exist_ok=True)
    json.dump({"gate": {"valid_candidates": len(valid), "pass_ge20": len(valid) >= 20,
                        "selected": len(selected)},
               "status_counts": {s: sum(1 for r in rows if r["status"] == s)
                                 for s in sorted({r["status"] for r in rows})},
               "audit_set": selected, "shortlist": short, "all_rows": rows},
              open(os.path.join(outdir, "audit_candidates.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(os.path.join(outdir, "review_shortlist.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "pilot_qid", "doc_id", "question", "gold_answer",
                    "gold_page", "swap_page", "swap_rule", "replacement", "benign_pages"])
        for i, r in enumerate(short, 1):
            w.writerow([f"cf_{i:03d}", r.get("pilot_qid", ""), r["doc_id"], r["question"],
                        r["answer"], r["gold_pages"][0], r["swap_page"], r["swap_rule"],
                        r["swap_replacement"], "|".join(map(str, r["benign_pages"]))])
    print(json.dumps({"gate": {"valid": len(valid), "pass": len(valid) >= 20,
                               "selected": len(selected), "shortlist": len(short)},
                      "status_counts": {s: sum(1 for r in rows if r["status"] == s)
                                        for s in sorted({r["status"] for r in rows})}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
