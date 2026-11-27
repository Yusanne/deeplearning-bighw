#!/usr/bin/env python
"""S17: evaluate a trained reranker checkpoint on the frozen eval splits.

The script accepts either an S16 output directory containing ``epoch*.pt`` or
an ordinary Hugging Face model directory.  MMLongBench-Doc and QASPER are
reported separately because their evidence units are different.

Usage:
  python -I s17_eval_reranker.py models/qasper_supervised_v1 \
      data_splits/eval_main80.json data/pages results/raw/qasper_model_eval \
      --answers results/raw/gen_answers_eval.jsonl --device cuda
"""
import argparse
import glob
import json
import os
import re
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


def eval_qid(dataset, split, row):
    if dataset == "mmlongbench_doc":
        return f"gen_ev::mm::{split}::{row['doc_id']}::{L.norm(row['question'])[:40]}"
    return f"gen_ev::qasper::{split}::{row['doc_id']}::{row['question_id']}"


def load_generated_answers(path):
    if not path:
        return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if not d.get("failed") and d.get("gen_answer"):
            out[d["qid"]] = str(d["gen_answer"])
    return out


def load_model(checkpoint, base_model, device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if os.path.isdir(checkpoint) and (
            os.path.exists(os.path.join(checkpoint, "config.json")) or
            os.path.exists(os.path.join(checkpoint, "model.safetensors")) or
            os.path.exists(os.path.join(checkpoint, "pytorch_model.bin"))):
        tok = AutoTokenizer.from_pretrained(checkpoint)
        net = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        resolved = checkpoint
    else:
        pt = checkpoint
        if os.path.isdir(checkpoint):
            candidates = glob.glob(os.path.join(checkpoint, "epoch*.pt"))
            if not candidates:
                raise FileNotFoundError(f"no epoch*.pt checkpoint in {checkpoint}")
            pt = max(candidates, key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
        tok = AutoTokenizer.from_pretrained(base_model)
        net = AutoModelForSequenceClassification.from_pretrained(base_model)
        state = torch.load(pt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state)
        resolved = pt
    net.to(device)
    net.eval()
    return net, tok, resolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("eval_path")
    ap.add_argument("pages_dir")
    ap.add_argument("outdir")
    ap.add_argument("--answers", default="",
                    help="optional gen_answers_eval.jsonl for the runtime gen slot")
    ap.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--max-questions", type=int, default=0,
                    help="smoke-test limit per split; 0 means all")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    net, tok, resolved = load_model(args.checkpoint, args.base_model, args.device)
    generated = load_generated_answers(args.answers)

    @torch.no_grad()
    def score(pairs):
        scores = []
        for i in range(0, len(pairs), args.batch_size):
            batch = tok(pairs[i:i + args.batch_size], padding=True,
                        truncation=True, max_length=args.max_length,
                        return_tensors="pt").to(args.device)
            scores.extend(net(**batch).logits.squeeze(-1).detach().cpu().tolist())
        return scores

    ev = json.load(open(args.eval_path, encoding="utf-8"))
    dataset = "qasper" if "qasper" in os.path.basename(args.eval_path) else "mmlongbench_doc"
    rows, agg = [], {}
    for split in ("dev", "test"):
        questions = ev.get(split) or ev.get("questions") or []
        if args.max_questions:
            questions = questions[:args.max_questions]
        bucket = {"bm25": [], "tfidf_proxy": [], "rrf": [],
                  "finetuned_none": [], "finetuned_gen": []}
        for r in questions:
            pages = L.load_pages(args.pages_dir, r["doc_id"])
            if not pages or not L.corpus_pages(pages):
                continue
            if ev.get("strict_evidence"):
                gold = [int(p) for p in r["evidence_pages"] if int(p) in pages]
                gold_src = "strict-field"
            else:
                gold, gold_src = L.gold_pages_by_answer(
                    pages, r["answer"], r.get("evidence_pages", []))
            bm, tf, rrf = L.retrieve(pages, r["question"])
            cand = rrf[:args.topk]
            q_none = f"Question: {r['question']}\nAnswer: [NO_ANSWER]"
            gen = generated.get(eval_qid(dataset, split, r), "[NO_ANSWER]")
            q_gen = f"Question: {r['question']}\nAnswer: {gen}"
            none_scores = score([(q_none, pages[p][:MAX_CHARS]) for p in cand])
            gen_scores = score([(q_gen, pages[p][:MAX_CHARS]) for p in cand])
            order_none = [cand[i] for i in sorted(range(len(cand)),
                                                    key=lambda j: -none_scores[j])]
            order_gen = [cand[i] for i in sorted(range(len(cand)),
                                                  key=lambda j: -gen_scores[j])]
            orders = {"bm25": bm, "tfidf_proxy": tf, "rrf": rrf,
                      "finetuned_none": order_none, "finetuned_gen": order_gen}
            metrics = {name: hit_metrics(order, gold) for name, order in orders.items()}
            for name in bucket:
                bucket[name].append(metrics[name])
            rows.append({"split": split, "dataset": dataset,
                         "qid": eval_qid(dataset, split, r),
                         "gold": gold, "gold_src": gold_src,
                         "generated_answer": gen,
                         "metrics": metrics,
                         "finetuned_none_top5": order_none[:5],
                         "finetuned_gen_top5": order_gen[:5]})
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
    result = {"checkpoint": resolved, "base_model": args.base_model,
              "eval_path": args.eval_path, "pages_dir": args.pages_dir,
              "answer_file": args.answers or None, "metrics": agg, "rows": rows,
              "note": "MMLongBench and QASPER are reported separately; tfidf_proxy is not dense."}
    with open(os.path.join(args.outdir, "reranker_eval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps({"checkpoint": resolved, "metrics": agg}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
