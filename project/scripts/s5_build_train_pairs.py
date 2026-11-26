#!/usr/bin/env python
"""S5: build reranker training pairs from the full corpus (doc-isolated from all eval/audit docs).

Pair = (question, answer_slot, evidence_text, label). Answer slots per frozen §3.6.2:
  gold — dataset gold answer (train pool only)
  none — no-answer support-scoring variant ("[NO_ANSWER]")
  gen  — filled later by s6_gen_train_answers (merged into train_pairs_final.jsonl)

Also emits gen_manifest.jsonl (question + top-5 context incl. gold page) for s6.

Usage: python -I s5_build_train_pairs.py <samples.json> <data_splits_dir> <audit_candidates.json> <pages_dir> <outdir>
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_cf as L

ALLOWED = {"Pure-text (Plain-text)", "Table"}
MAX_CHARS = 4000
HARD_NEG_K = 8


def main(samples_path, splits_dir, audit_path, pages_dir, outdir):
    audit_docs = {r["doc_id"] for r in json.load(open(audit_path, encoding="utf-8"))["audit_set"]}
    # Blocking policy (doc-level isolation):
    #  - eval_main80 docs: hard block (dev thresholds + test metrics; no train leakage);
    #  - selected audit docs: block (audit layer must stay unseen);
    #  - eval_unans40 docs: ALLOWED — that layer measures abstention, not page ranking;
    #    reranker training on their answerable questions leaks no page-ranking labels.
    eval_docs = set()
    d = json.load(open(os.path.join(splits_dir, "eval_main80.json"), encoding="utf-8"))
    eval_docs |= {q["doc_id"] for q in d["questions"]}
    data = json.load(open(samples_path, encoding="utf-8"))
    items = data["samples"] if isinstance(data, dict) and "samples" in data else data

    blocked = eval_docs | audit_docs
    pairs, manifest, skipped = [], [], Counter()
    for it in items:
        doc = it["doc_id"]
        if doc in blocked:
            skipped["eval_or_audit_doc"] += 1
            continue
        srcs = [str(s) for s in L.parse_maybe_list(it.get("evidence_sources", "[]"))]
        if not srcs or not all(s in ALLOWED for s in srcs):
            skipped["modality"] += 1
            continue
        fmt = str(it.get("answer_format", "")).strip().lower()
        ans = str(it.get("answer", "")).strip()
        if fmt.startswith("null") or fmt in ("none", "") or not ans:
            skipped["unanswerable"] += 1
            continue
        pages = L.load_pages(pages_dir, doc)
        if pages is None:
            skipped["no_pages"] += 1
            continue
        corpus = L.corpus_pages(pages)
        if not corpus:
            skipped["no_text"] += 1
            continue
        gold, src = L.gold_pages_by_answer(pages, ans, [int(p) for p in
                                                        L.parse_maybe_list(it.get("evidence_pages", "[]"))
                                                        if str(p).lstrip("-").isdigit()])
        if src != "answer-string" or not gold:
            skipped["gold_not_located"] += 1
            continue
        bm, tf, rrf = L.retrieve(pages, it["question"])
        negs = [p for p in rrf[:HARD_NEG_K + len(gold)] if p not in gold][:HARD_NEG_K]
        if not negs:
            skipped["no_negatives"] += 1
            continue
        qid = f"tr_{len(pairs):06d}"
        base = {"qid": qid, "doc_id": doc, "question": it["question"], "gold_answer": ans,
                "gold_pages": gold, "neg_pages": negs}
        for slot_key, slot_val in (("gold", ans), ("none", "[NO_ANSWER]")):
            for p in gold:
                pairs.append({**base, "answer_slot": slot_key, "answer_text": slot_val,
                              "page": p, "label": 1, "page_text": pages[p][:MAX_CHARS]})
            for p in negs:
                pairs.append({**base, "answer_slot": slot_key, "answer_text": slot_val,
                              "page": p, "label": 0, "page_text": pages[p][:MAX_CHARS]})
        ctx_pages = list(dict.fromkeys(gold + rrf[:5]))
        manifest.append({**base, "context_pages": ctx_pages,
                         "context_text": "\n\n".join(f"[Page {p}]\n{pages[p][:MAX_CHARS]}"
                                                     for p in ctx_pages)})

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "train_pairs.jsonl"), "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(outdir, "gen_manifest.jsonl"), "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    stats = {"questions": len(manifest), "pairs": len(pairs),
             "pos": sum(1 for p in pairs if p["label"] == 1),
             "neg": sum(1 for p in pairs if p["label"] == 0),
             "blocked_docs": len(blocked), "skipped": dict(skipped)}
    json.dump(stats, open(os.path.join(outdir, "train_pairs_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
