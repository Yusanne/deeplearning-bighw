#!/usr/bin/env python
"""Week1 S3: extract per-page text from selected PDFs and audit answer/evidence alignment.

Usage: python -I s3_extract_audit.py <selected20.json> <documents_dir> <pages_dir> <out_audit.json>
"""
import json
import os
import re
import sys

import pymupdf


def norm(s):
    return re.sub(r"[%$€£'\"\s(),.]+", "", str(s)).lower()


def main(selected_path, docs_dir, pages_dir, out_path):
    sel = json.load(open(selected_path, encoding="utf-8"))
    docs = {}
    for q in sel["questions"]:
        if q["doc_id"] in docs:
            continue
        path = os.path.join(docs_dir, q["doc_id"])
        doc = pymupdf.open(path)
        pages = {}
        poor = 0
        for i in range(doc.page_count):
            t = doc[i].get_text() or ""
            pages[str(i)] = t
            if len(t.strip()) < 50:
                poor += 1
        docs[q["doc_id"]] = {"n_pages": doc.page_count, "poor_pages": poor, "pages": pages}
        doc.close()
        with open(os.path.join(pages_dir, q["doc_id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False)

    audit = []
    for q in sel["questions"]:
        d = docs[q["doc_id"]]
        ev_pages = q["evidence_pages"]
        ev = ev_pages[0] if ev_pages else None
        g = norm(q["answer"])
        found_on_ev = None
        if ev is not None:
            for cand in (ev, ev - 1, ev + 1):
                if 0 <= cand < d["n_pages"] and g and g in norm(d["pages"][str(cand)]):
                    found_on_ev = cand
                    break
        ev_len = len(d["pages"][str(ev)].strip()) if ev is not None and 0 <= ev < d["n_pages"] else 0
        ans_in_doc = any(g in norm(t) for t in d["pages"].values()) if g else False
        audit.append({
            "qid": q["qid"], "doc_id": q["doc_id"], "n_pages": d["n_pages"],
            "poor_pages": d["poor_pages"], "ev_page_raw": ev, "ev_text_len": ev_len,
            "answer_on_page": found_on_ev, "idx_base_if_found": (
                "1-based" if found_on_ev == ev - 1 else
                "0-based" if found_on_ev == ev else
                "neighbor" if found_on_ev is not None else "not-found"),
            "answer_in_doc": ans_in_doc,
            "ev_extractable": ev_len >= 200,
        })

    n_ok_ev = sum(1 for a in audit if a["ev_extractable"])
    n_found = sum(1 for a in audit if a["answer_on_page"] is not None)
    n_indoc = sum(1 for a in audit if a["answer_in_doc"])
    bases = {}
    for a in audit:
        b = a["idx_base_if_found"]
        bases[b] = bases.get(b, 0) + 1
    summary = {
        "questions": len(audit),
        "evidence_page_text_extractable(>=200chars)": n_ok_ev,
        "answer_found_on_evidence_page": n_found,
        "answer_found_anywhere_in_doc": n_indoc,
        "idx_base_distribution": bases,
        "total_docs": len(docs),
        "total_poor_pages": sum(d["poor_pages"] for d in docs.values()),
    }
    json.dump({"summary": summary, "audit": audit}, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
