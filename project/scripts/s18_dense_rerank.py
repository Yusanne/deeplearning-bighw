#!/usr/bin/env python
"""S18: dense retrieval baseline with sentence-transformers/all-MiniLM-L6-v2.

This is an inference-only baseline.  It embeds every page in a document and
the question with the same encoder, then reports Dense and BM25+Dense RRF
rankings.  The implementation uses ``transformers`` directly so the project
does not depend on the sentence-transformers package at runtime.

Usage:
  python -I s18_dense_rerank.py data_splits/eval_main80.json data/pages \
      results/raw/dense_mm_eval --device cuda
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_cf as L


TOPK = 20
MAX_CHARS = 3500


def hit_metrics(order, gold):
    h1 = int(any(p in gold for p in order[:1]))
    h5 = int(any(p in gold for p in order[:5]))
    mrr = next((1.0 / i for i, p in enumerate(order, 1) if p in gold), 0.0)
    return h1, h5, mrr


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode(texts, tokenizer, model, device, max_length, batch_size):
    import torch

    out = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], padding=True,
                          truncation=True, max_length=max_length,
                          return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
            vec = mean_pool(hidden, batch["attention_mask"])
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        out.append(vec.cpu())
    return torch.cat(out, dim=0) if out else torch.empty((0, model.config.hidden_size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_path")
    ap.add_argument("pages_dir")
    ap.add_argument("outdir")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--max-questions", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device)
    model.eval()

    ev = json.load(open(args.eval_path, encoding="utf-8"))
    rows, agg = [], {}
    page_cache = {}
    for split in ("dev", "test"):
        questions = ev.get(split) or ev.get("questions") or []
        if args.max_questions:
            questions = questions[:args.max_questions]
        bucket = {"bm25": [], "tfidf_proxy": [], "rrf_tfidf": [],
                  "dense": [], "rrf_dense": []}
        for idx, r in enumerate(questions, 1):
            pages = L.load_pages(args.pages_dir, r["doc_id"])
            if not pages or not L.corpus_pages(pages):
                continue
            if ev.get("strict_evidence"):
                gold = [int(p) for p in r["evidence_pages"] if int(p) in pages]
                gold_src = "strict-field"
            else:
                gold, gold_src = L.gold_pages_by_answer(
                    pages, r["answer"], r.get("evidence_pages", []))
            corpus = L.corpus_pages(pages)
            cache_key = (args.pages_dir, r["doc_id"])
            if cache_key not in page_cache:
                texts = [pages[p][:MAX_CHARS] for p in corpus]
                page_cache[cache_key] = (corpus, encode(
                    texts, tokenizer, model, args.device, args.max_length,
                    args.batch_size))
            cached_pages, page_vecs = page_cache[cache_key]
            qvec = encode([r["question"]], tokenizer, model, args.device,
                          args.max_length, args.batch_size)[0]
            dense_scores = page_vecs @ qvec
            dense = [cached_pages[i] for i in torch.argsort(dense_scores, descending=True).tolist()]
            bm, tf, rrf_tfidf = L.retrieve(pages, r["question"])
            rrf_dense = L.rrf_fuse([bm, dense])
            orders = {"bm25": bm, "tfidf_proxy": tf, "rrf_tfidf": rrf_tfidf,
                      "dense": dense, "rrf_dense": rrf_dense}
            metrics = {name: hit_metrics(order, gold) for name, order in orders.items()}
            for name in bucket:
                bucket[name].append(metrics[name])
            rows.append({"split": split, "dataset": "qasper" if "qasper" in os.path.basename(args.eval_path) else "mmlongbench_doc",
                         "qid": f"{r['doc_id']}|{r['question'][:40]}",
                         "gold": gold, "gold_src": gold_src, "metrics": metrics,
                         "dense_top5": dense[:5], "rrf_dense_top5": rrf_dense[:5]})
            if idx % 10 == 0:
                print(f"{split}: {idx}/{len(questions)} questions", flush=True)
        agg[split] = {
            name: {"hit1": round(sum(x[0] for x in vals) / max(len(vals), 1), 3),
                   "hit5": round(sum(x[1] for x in vals) / max(len(vals), 1), 3),
                   "mrr": round(sum(x[2] for x in vals) / max(len(vals), 1), 3),
                   "n": len(vals)}
            for name, vals in bucket.items()
        }
        if "questions" in ev and "dev" not in ev:
            break

    os.makedirs(args.outdir, exist_ok=True)
    result = {"model": args.model, "eval_path": args.eval_path,
              "pages_dir": args.pages_dir, "metrics": agg, "rows": rows,
              "note": "Dense mean-pooling baseline; rrf_dense fuses BM25 and Dense rankings."}
    with open(os.path.join(args.outdir, "dense_rerank.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps({"model": args.model, "metrics": agg}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
