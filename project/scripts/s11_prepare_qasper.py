#!/usr/bin/env python
"""Prepare a small, evidence-preserving QASPER external-validation slice.

QASPER evidence is paragraph-level rather than PDF-page-level.  We therefore
keep it as paragraph units and never call it a page-level result.  The output
is intentionally small enough to run the same retrieval/reranker harness.

Usage:
  python -I s11_prepare_qasper.py <raw_train.json> <raw_dev.json> <raw_test.json>
      <pages_dir> <eval_json> <stats_json>
"""
import json
import os
import re
import sys
from collections import Counter


def clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def flatten(paper):
    out = []
    for sec in paper.get("full_text", []):
        name = clean(sec.get("section_name", ""))
        for para in sec.get("paragraphs", []):
            text = clean(para)
            if text:
                out.append(f"[{name}] {text}" if name else text)
    return out


def answer_text(ans):
    if ans.get("unanswerable"):
        return ""
    if ans.get("free_form_answer"):
        return clean(ans["free_form_answer"])
    if ans.get("yes_no") is not None:
        return "yes" if ans["yes_no"] else "no"
    spans = ans.get("extractive_spans") or []
    return clean(" ".join(spans))


def evidence_indices(paras, evidence):
    idx = []
    normed = [clean(x).lower() for x in paras]
    for ev in evidence or []:
        e = clean(ev).lower()
        if not e:
            continue
        hits = [i for i, p in enumerate(normed) if e == p or e in p or p in e]
        if hits:
            idx.append(hits[0])
    return sorted(set(idx))


def load(path):
    return json.load(open(path, encoding="utf-8"))


def main(train_path, dev_path, test_path, pages_dir, eval_path, stats_path):
    all_counts = {}
    records = []
    for split, path in (("train", train_path), ("dev", dev_path), ("test", test_path)):
        data = load(path)
        q_total = q_answerable = q_with_evidence = 0
        for paper_id, paper in data.items():
            paras = flatten(paper)
            if split != "train":
                with open(os.path.join(pages_dir, f"qasper_{paper_id}.json"), "w", encoding="utf-8") as f:
                    json.dump({str(i): p for i, p in enumerate(paras)}, f, ensure_ascii=False)
            for qa in paper.get("qas", []):
                q_total += 1
                ann = next((a.get("answer", {}) for a in qa.get("answers", [])
                            if not a.get("answer", {}).get("unanswerable")), None)
                if not ann:
                    continue
                q_answerable += 1
                ans = answer_text(ann)
                ev_pages = evidence_indices(paras, ann.get("evidence", []))
                if ev_pages:
                    q_with_evidence += 1
                if split == "dev" and ans and ev_pages:
                    records.append({
                        "doc_id": f"qasper_{paper_id}",
                        "paper_id": paper_id,
                        "question": clean(qa.get("question", "")),
                        "answer": ans,
                        "evidence_pages": ev_pages,
                        "evidence_sources": ["QASPER paragraph evidence"],
                        "src_modality": "qasper_paragraph",
                        "question_id": qa.get("question_id", ""),
                    })
        all_counts[split] = {"papers": len(data), "questions": q_total,
                             "answerable_questions": q_answerable,
                             "questions_with_mapped_evidence": q_with_evidence}

    # Fixed, document-diverse 40-question external slice.  It is explicitly
    # a paragraph-level robustness check, not a replacement for the PDF task.
    records.sort(key=lambda r: (r["doc_id"], r["question_id"]))
    chosen, seen_docs = [], set()
    for r in records:
        if r["doc_id"] in seen_docs:
            continue
        chosen.append(r)
        seen_docs.add(r["doc_id"])
        if len(chosen) >= 40:
            break
    if len(chosen) < 40:
        for r in records:
            if r in chosen:
                continue
            chosen.append(r)
            if len(chosen) >= 40:
                break
    mid = len(chosen) // 2
    out = {"count": len(chosen), "dev": chosen[:mid], "test": chosen[mid:],
           "source": "QASPER v0.3 official evidence annotations",
           "unit": "paragraph (not PDF page)",
           "strict_evidence": True}
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    json.dump(out, open(eval_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    stats = {"source": "QASPER v0.3 official S3 release", "splits": all_counts,
             "external_slice": {"questions": len(chosen),
                                "documents": len({r["doc_id"] for r in chosen}),
                                "dev": len(out["dev"]), "test": len(out["test"]),
                                "unit": "paragraph"}}
    json.dump(stats, open(stats_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
