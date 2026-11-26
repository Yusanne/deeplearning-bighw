#!/usr/bin/env python
"""Build evidence-selection pairs from the official QASPER train release.

Evidence units remain paragraphs.  This is the scalable supervision source for
the formal reranker run; it must not be merged with page-level MMLongBench
metrics as if the units were identical.
"""
import json
import os
import re
import sys


def clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def toks(s):
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


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
    return clean(" ".join(ans.get("extractive_spans") or []))


def evidence_indices(paras, evidence):
    out = []
    normed = [clean(x).lower() for x in paras]
    for ev in evidence or []:
        e = clean(ev).lower()
        hits = [i for i, p in enumerate(normed) if e == p or e in p or p in e]
        if hits:
            out.append(hits[0])
    return sorted(set(out))


def main(raw_path, out_path, stats_path):
    data = json.load(open(raw_path, encoding="utf-8"))
    rows = []
    questions = 0
    skipped = {"unanswerable": 0, "no_evidence_mapping": 0, "no_negatives": 0}
    for paper_id, paper in data.items():
        paras = flatten(paper)
        para_tokens = [toks(p) for p in paras]
        for qa in paper.get("qas", []):
            questions += 1
            ann = next((a.get("answer", {}) for a in qa.get("answers", [])
                        if not a.get("answer", {}).get("unanswerable")), None)
            if not ann:
                skipped["unanswerable"] += 1
                continue
            gold = evidence_indices(paras, ann.get("evidence", []))
            if not gold:
                skipped["no_evidence_mapping"] += 1
                continue
            q = clean(qa.get("question", ""))
            ans = answer_text(ann)
            qt = toks(q)
            scored = []
            for i, pt in enumerate(para_tokens):
                if i in gold:
                    continue
                score = len(qt & pt) / max(1, len(qt))
                scored.append((score, -i, i))
            scored.sort(reverse=True)
            negs = [i for _, _, i in scored[:8]]
            if not negs:
                skipped["no_negatives"] += 1
                continue
            base = {"qid": f"qasper_{paper_id}_{qa.get('question_id','')}",
                    "doc_id": f"qasper_{paper_id}", "question": q,
                    "gold_answer": ans, "gold_pages": gold, "neg_pages": negs,
                    "source_unit": "paragraph"}
            for slot, val in (("gold", ans), ("none", "[NO_ANSWER]")):
                for p in gold:
                    rows.append({**base, "answer_slot": slot, "answer_text": val,
                                 "page": p, "label": 1,
                                 "page_text": paras[p][:4000]})
                for p in negs:
                    rows.append({**base, "answer_slot": slot, "answer_text": val,
                                 "page": p, "label": 0,
                                 "page_text": paras[p][:4000]})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"questions_seen": questions,
             "questions_with_pairs": len({r["qid"] for r in rows}),
             "pairs": len(rows), "positive": sum(r["label"] == 1 for r in rows),
             "negative": sum(r["label"] == 0 for r in rows), "skipped": skipped,
             "source": "QASPER v0.3 official train", "unit": "paragraph"}
    json.dump(stats, open(stats_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
