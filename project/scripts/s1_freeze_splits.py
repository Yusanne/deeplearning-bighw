#!/usr/bin/env python
"""S1: freeze evaluation splits from MMLongBench-Doc samples.json (frozen protocol §7).

Outputs into data_splits/:
  eval_main80.json   — 80 answerable (Pure-text+Table), stratified, dev20/test60 doc-isolated
  eval_unans40.json  — 40 native unanswerable, doc-disjoint from main80
  audit_pool.json    — candidate universe for CiteFaith audit autofilter
  splits_summary.json

Usage: python -I s1_freeze_splits.py <samples.json> <outdir> [pilot_selected20.json]
"""
import ast
import json
import random
import re
import sys
from collections import Counter, defaultdict

SEED = 2024201433
CAP_PER_DOC = 3
PURE = "Pure-text (Plain-text)"
TABLE = "Table"
ALLOWED = {PURE, TABLE}


def parse_maybe_list(v):
    if isinstance(v, list):
        return v
    try:
        x = ast.literal_eval(v) if isinstance(v, str) else [v]
        return x if isinstance(x, list) else [x]
    except Exception:
        return [v]


def norm(s):
    return re.sub(r"[%$€£'\"\s(),.]+", "", str(s)).lower()


def answerable(it):
    fmt = str(it.get("answer_format", "")).strip().lower()
    if fmt.startswith("null") or fmt in ("none", ""):
        return False
    if norm(it.get("answer", "")) in ("", "notanswerable", "unanswerable", "noanswer", "na"):
        return False
    return True


def rec(it, src_modality):
    return {
        "doc_id": it["doc_id"],
        "doc_type": it.get("doc_type", ""),
        "question": it["question"],
        "answer": str(it.get("answer", "")).strip(),
        "answer_format": str(it.get("answer_format")),
        "evidence_pages": [int(p) for p in parse_maybe_list(it.get("evidence_pages", "[]"))
                           if str(p).strip().lstrip("-").isdigit()],
        "evidence_sources": [str(s) for s in parse_maybe_list(it.get("evidence_sources", "[]"))],
        "src_modality": src_modality,
    }


def stratified_sample(pool, n, rng, cap=CAP_PER_DOC):
    bykey = defaultdict(list)
    for r in pool:
        key = (r["answer_format"], r["src_modality"])
        bykey[key].append(r)
    total = len(pool)
    alloc = {}
    acc = 0
    keys = sorted(bykey, key=lambda k: -len(bykey[k]))
    for i, k in enumerate(keys):
        take = round(len(bykey[k]) / total * n) if i < len(keys) - 1 else n - acc
        alloc[k] = max(0, take)
        acc += alloc[k]
    picked, per_doc = [], Counter()
    for k in keys:
        cand = bykey[k][:]
        rng.shuffle(cand)
        for r in cand:
            if len([p for p in picked if (p["answer_format"], p["src_modality"]) == k]) >= alloc[k]:
                continue
            if per_doc[r["doc_id"]] >= cap:
                continue
            picked.append(r)
            per_doc[r["doc_id"]] += 1
    # top-up ignoring strata if short
    if len(picked) < n:
        rest = [r for r in pool if r not in picked]
        rng.shuffle(rest)
        for r in rest:
            if len(picked) >= n:
                break
            if per_doc[r["doc_id"]] >= cap:
                continue
            picked.append(r)
            per_doc[r["doc_id"]] += 1
    return picked


def main(samples_path, outdir, pilot_path=None):
    data = json.load(open(samples_path, encoding="utf-8"))
    items = data["samples"] if isinstance(data, dict) and "samples" in data else data
    pilot_qs, pilot_docs = {}, set()
    if pilot_path:
        for q in json.load(open(pilot_path, encoding="utf-8"))["questions"]:
            pilot_qs[(q["doc_id"], q["question"])] = q["qid"]
            pilot_docs.add(q["doc_id"])

    elig, unans = [], []
    for it in items:
        srcs = [str(s) for s in parse_maybe_list(it.get("evidence_sources", "[]"))]
        if not answerable(it):
            unans.append(it)
            continue
        if not srcs or not all(s in ALLOWED for s in srcs):
            continue
        if (it["doc_id"], it["question"]) in pilot_qs or it["doc_id"] in pilot_docs:
            continue  # pilot docs are reserved for the audit layer
        mod = "pure" if all(s == PURE for s in srcs) else "table"
        elig.append(rec(it, mod))

    rng = random.Random(SEED)
    main80 = stratified_sample(elig, 80, rng)
    main_ids = {id(r) for r in main80}
    main_docs = {r["doc_id"] for r in main80}

    # dev/test doc-isolated split: greedily fill dev20 by whole docs
    docs = defaultdict(list)
    for r in main80:
        docs[r["doc_id"]].append(r)
    doc_names = sorted(docs)
    rng.shuffle(doc_names)
    dev, test, ndev = [], [], 0
    for d in doc_names:
        if ndev < 20:
            dev.extend(docs[d])
            ndev += len(docs[d])
        else:
            test.extend(docs[d])

    # audit pool FIRST: pilot docs + pure-text short answers from non-eval docs
    audit_pool = []
    for it in items:
        if not answerable(it):
            continue
        srcs = [str(s) for s in parse_maybe_list(it.get("evidence_sources", "[]"))]
        if not srcs or not all(s == PURE for s in srcs):
            continue
        if it["doc_id"] in main_docs:
            continue
        ans = str(it.get("answer", "")).strip()
        if not (1 <= len(ans) <= 40):
            continue
        r2 = rec(it, "pure")
        r2["pref_single_evidence"] = len(r2["evidence_pages"]) == 1
        r2["pilot_qid"] = pilot_qs.get((it["doc_id"], it["question"]), "")
        audit_pool.append(r2)
    audit_docs = {r["doc_id"] for r in audit_pool}

    # unanswerable 40: doc-disjoint from main80 AND audit pool
    unans_pool = [rec(it, "unans") for it in unans
                  if it["doc_id"] not in main_docs and it["doc_id"] not in audit_docs]
    uns40 = stratified_sample(unans_pool, 40, rng)
    uns_docs = {r["doc_id"] for r in uns40}

    def dump(name, rows, extra=None):
        out = {"seed": SEED, "count": len(rows), **(extra or {}), "questions": rows}
        json.dump(out, open(f"{outdir}/{name}", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    dump("eval_main80.json", main80,
         {"dev": [r for r in dev], "test": [r for r in test],
          "strata": {f"{k[0]}|{k[1]}": v for k, v in
                     Counter((r["answer_format"], r["src_modality"]) for r in main80).items()}})
    dump("eval_unans40.json", uns40)
    dump("audit_pool.json", audit_pool,
         {"n_docs": len({r['doc_id'] for r in audit_pool}),
          "n_single": sum(1 for r in audit_pool if r["pref_single_evidence"]),
          "n_pilot": sum(1 for r in audit_pool if r["pilot_qid"])})

    summary = {
        "eligible_pool": len(elig),
        "eligible_by_modality": dict(Counter(r["src_modality"] for r in elig)),
        "main80": len(main80), "main80_docs": len(main_docs),
        "dev": len(dev), "test": len(test),
        "dev_test_doc_isolated": True,
        "unanswerable_total": len(unans),
        "unans40": len(uns40), "unans40_docs": len(uns_docs),
        "audit_pool": len(audit_pool),
        "audit_pool_docs": len({r["doc_id"] for r in audit_pool}),
        "audit_pool_single_evidence": sum(1 for r in audit_pool if r["pref_single_evidence"]),
        "audit_pool_pilot_overlap": sum(1 for r in audit_pool if r["pilot_qid"]),
    }
    json.dump(summary, open(f"{outdir}/splits_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
