#!/usr/bin/env python
"""S15: build two generation-answer manifests for the frozen answer-slot protocol.

A) train-side (gold evidence in context): QASPER train questions with pairs + MM train pool.
   Used ONLY as the 'gen' answer-slot training variant (§3.6.2).
B) eval-side (RRF top-5 retrieved context, no gold): MM dev+test, QASPER dev+test.
   Used as the runtime answer for B5a/B5b scoring and generation baselines.

Usage: python -I s15_build_gen_manifests.py <outdir>
"""
import json
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_cf as L

MAX_CTX_CHARS = 6000


def main(outdir):
    os.makedirs(outdir, exist_ok=True)

    # ---- A) train-side: gold evidence in context ----
    seen, train_rows = set(), []
    for line in open("results/raw/mixed_train_pairs.jsonl", encoding="utf-8"):
        r = json.loads(line)
        key = (r["source_dataset"], r["doc_id"], r["question"])
        if key in seen:
            continue
        seen.add(key)
        ctx = r["page_text"][:MAX_CTX_CHARS]
        if not ctx:
            continue
        train_rows.append({"qid": "",  # filled below, stable per question
                           "source_dataset": r["source_dataset"], "doc_id": r["doc_id"],
                           "question": r["question"], "context_text": ctx})
    for i, row in enumerate(train_rows):
        row["qid"] = f"trq_{i:05d}"
    with open(os.path.join(outdir, "gen_manifest_train.jsonl"), "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- B) eval-side: retrieved top-5, no gold ----
    eval_rows = []
    ev = json.load(open("data_splits/eval_main80.json", encoding="utf-8"))
    for split in ("dev", "test"):
        for r in ev[split]:
            pages = L.load_pages("data/pages", r["doc_id"])
            if pages is None or not L.corpus_pages(pages):
                continue
            _, _, rrf = L.retrieve(pages, r["question"])
            ctx = "\n\n".join(f"[Page {p}]\n{pages[p][:1500]}" for p in rrf[:5])[:MAX_CTX_CHARS]
            eval_rows.append({"qid": f"gen_ev::mm::{split}::{r['doc_id']}::{L.norm(r['question'])[:40]}",
                              "dataset": "mmlongbench_doc", "split": split,
                              "doc_id": r["doc_id"], "question": r["question"],
                              "context_text": ctx})
    qv = json.load(open("data_splits/eval_qasper40.json", encoding="utf-8"))
    for split in ("dev", "test"):
        for r in qv[split]:
            pages = L.load_pages("data/qasper_pages", r["doc_id"])
            if pages is None:
                continue
            corpus = L.corpus_pages(pages)
            if not corpus:
                continue
            bm, tf, rrf = L.retrieve(pages, r["question"])
            ctx = "\n\n".join(f"[Paragraph {p}]\n{pages[p][:1500]}" for p in rrf[:5])[:MAX_CTX_CHARS]
            eval_rows.append({"qid": f"gen_ev::qasper::{split}::{r['doc_id']}::{r['question_id']}",
                              "dataset": "qasper", "split": split,
                              "doc_id": r["doc_id"], "question": r["question"],
                              "context_text": ctx})
    with open(os.path.join(outdir, "gen_manifest_eval.jsonl"), "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({"train_questions": len(train_rows),
                      "eval_questions": len(eval_rows),
                      "eval_by": {"mm": sum(1 for r in eval_rows if r["dataset"].startswith("mm")),
                                   "qasper": sum(1 for r in eval_rows if r["dataset"] == "qasper")}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/raw")
