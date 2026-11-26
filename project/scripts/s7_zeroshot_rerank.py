#!/usr/bin/env python
"""S7: zero-shot reranker scoring on eval candidates (public Cross-Encoder, no training).
Scores RRF top-20 candidates per eval question with two answer slots (oracle-gold / none) and
reports Hit@1/Hit@5/MRR for BM25 / TF-IDF / RRF / ZeroShot(oracle) / ZeroShot(no-answer).

NOTE: TF-IDF stands in for Dense until the DashScope text-embedding pass is wired in; label
outputs accordingly (dense_tfidf_proxy).

Env: RERANK_MODEL (default cross-encoder/ms-marco-MiniLM-L6-v2), HF_ENDPOINT for mirrors.
Usage: python -I s7_zeroshot_rerank.py <eval_main80.json> <pages_dir> <outdir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_cf as L

TOPK = 20
MAX_CHARS = 3500


def hit_metrics(order, gold):
    def hit(k):
        return int(any(p in gold for p in order[:k]))
    mrr = 0.0
    for i, p in enumerate(order, 1):
        if p in gold:
            mrr = 1.0 / i
            break
    return hit(1), hit(5), mrr


def main(eval_path, pages_dir, outdir):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_name = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
    tok = AutoTokenizer.from_pretrained(model_name)
    net = AutoModelForSequenceClassification.from_pretrained(model_name)
    net.eval()

    @torch.no_grad()
    def score(pairs):
        out = []
        B = 16
        for i in range(0, len(pairs), B):
            batch = tok(pairs[i:i + B], padding=True, truncation=True, max_length=512,
                        return_tensors="pt")
            logits = net(**batch).logits.squeeze(-1)
            out.extend(torch.sigmoid(logits).tolist())
        return out

    ev = json.load(open(eval_path, encoding="utf-8"))
    rows, agg = [], {}
    for split in ("dev", "test"):
        qs = ev.get(split) or ev["questions"]
        m = {"bm25": [], "tfidf": [], "rrf": [], "zs_oracle": [], "zs_none": []}
        for r in qs:
            pages = L.load_pages(pages_dir, r["doc_id"])
            corpus = L.corpus_pages(pages)
            if ev.get("strict_evidence"):
                gold = [int(p) for p in r["evidence_pages"] if int(p) in pages]
                src = "strict-field"
            else:
                gold, src = L.gold_pages_by_answer(pages, r["answer"], r["evidence_pages"])
            bm, tf, rrf = L.retrieve(pages, r["question"])
            cand = rrf[:TOPK]
            q_oracle = f"Question: {r['question']}\nAnswer: {r['answer']}"
            q_none = f"Question: {r['question']}\nAnswer: [NO_ANSWER]"
            s_or = score([(q_oracle, pages[p][:MAX_CHARS]) for p in cand])
            s_no = score([(q_none, pages[p][:MAX_CHARS]) for p in cand])
            zs_or = [cand[i] for i in sorted(range(len(cand)), key=lambda j: -s_or[j])]
            zs_no = [cand[i] for i in sorted(range(len(cand)), key=lambda j: -s_no[j])]
            for name, order in (("bm25", bm), ("tfidf", tf), ("rrf", rrf),
                                ("zs_oracle", zs_or), ("zs_none", zs_no)):
                m[name].append(hit_metrics(order, gold))
            rows.append({"split": split,
                         "qid": r.get("doc_id", "")[:20] + "|" + r["question"][:40],
                         "gold": gold, "gold_src": src,
                         "metrics": {name: hit_metrics(order, gold) for name, order in
                                     (("bm25", bm), ("tfidf", tf), ("rrf", rrf),
                                      ("zs_oracle", zs_or), ("zs_none", zs_no))},
                         "zs_oracle_top5": zs_or[:5], "zs_none_top5": zs_no[:5]})
        agg[split] = {name: {"hit1": round(sum(x[0] for x in v) / len(v), 3),
                             "hit5": round(sum(x[1] for x in v) / len(v), 3),
                             "mrr": round(sum(x[2] for x in v) / len(v), 3)}
                      for name, v in m.items()}
        if "questions" in ev and "dev" not in ev:
            break
    os.makedirs(outdir, exist_ok=True)
    json.dump({"model": model_name, "note": "tfidf = dense proxy until embedding API wired",
               "metrics": agg, "rows": rows},
              open(os.path.join(outdir, "zeroshot_rerank.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps({"model": model_name, "metrics": agg}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
