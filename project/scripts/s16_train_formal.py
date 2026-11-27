#!/usr/bin/env python
"""S16: formal reranker training with leakage-safe answer slots.

The input pair files may already contain repeated ``gold``/``none`` rows (the
QASPER builder does this), so rows are first deduplicated by
question-document-page-label.  Only then are the available answer slots
{gold, gen, none} expanded.  This keeps the reported training size honest.

Evaluation on MM dev + QASPER dev uses the *none* slot only (never gold at
evaluation).

Class imbalance is handled with a positive-class weight in BCE loss.

Usage:
  python -I s16_train_formal.py --pairs results/raw/mixed_train_pairs.jsonl \
      --gen results/raw/gen_answers_train.jsonl --out models/formal_v1 \
      --epochs 2 --max-length 512 --device cpu

Smoke test:
  python -I s16_train_formal.py --pairs results/raw/qasper_train_pairs.jsonl \
      --source-filter qasper --max-questions 2 --epochs 1 \
      --max-length 64 --batch-size 2 --skip-eval --out /tmp/s16-smoke
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
EVAL_SPLITS = {"mm": ("data_splits/eval_main80.json", "data/pages"),
               "qasper": ("data_splits/eval_qasper40.json", "data/qasper_pages")}


class PairDataset(Dataset):
    def __init__(self, rows, tok, max_length):
        self.rows, self.tok, self.max_length = rows, tok, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["query"], r["page_text"], truncation=True, max_length=self.max_length,
                       padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(float(r["label"]))


def load_pairs(pairs_path, gen_path, manifest_path, source_filter="", max_questions=0):
    def norm_ans(x):
        if isinstance(x, list):
            return " ".join(map(str, x))
        return str(x) if x is not None else ""
    gen_by_key = {}
    if os.path.exists(gen_path):
        answers = {}
        for line in open(gen_path, encoding="utf-8"):
            d = json.loads(line)
            answers[d["qid"]] = norm_ans(d.get("gen_answer", ""))
        for line in open(manifest_path, encoding="utf-8"):
            m = json.loads(line)
            a = answers.get(m["qid"], "")
            if a:
                gen_by_key[(m["source_dataset"], m["doc_id"], m["question"])] = a
    # A QASPER row already carries an answer_slot.  Do not treat the same
    # page/label under gold and none as two evidence candidates.
    by_q = defaultdict(lambda: {"base": None, "candidates": {}})
    n_skip = 0
    for line in open(pairs_path, encoding="utf-8"):
        r = json.loads(line)
        src = r.get("source_dataset") or "qasper"  # qasper-only file lacks the field
        if source_filter and src != source_filter:
            n_skip += 1
            continue
        key = (src, r["doc_id"], r["question"])
        bucket = by_q[key]
        if bucket["base"] is None:
            bucket["base"] = r
        page_key = (r.get("page"), int(r["label"]))
        bucket["candidates"].setdefault(page_key, r)
    if n_skip:
        print(f"source-filter={source_filter} skipped={n_skip}", flush=True)
    keys = list(by_q)
    if max_questions:
        keys = keys[:max_questions]
        print(f"max-questions={max_questions} selected={len(keys)}", flush=True)
    rows, slots = [], defaultdict(int)
    for key in keys:
        g = by_q[key]
        base = g["base"]
        variants = {"gold": base["gold_answer"], "none": "[NO_ANSWER]"}
        ga = gen_by_key.get(key, "")
        if ga:
            variants["gen"] = ga
        candidates = list(g["candidates"].values())
        pos = [r for r in candidates if r["label"] == 1]
        neg = [r for r in candidates if r["label"] == 0]
        for slot, atext in variants.items():
            for r in pos + neg:
                rows.append({"query": f"Question: {base['question']}\nAnswer: {atext}",
                             "page_text": r["page_text"], "label": r["label"]})
                slots[slot] += 1
    return rows, slots, len(keys)


@torch.no_grad()
def evaluate(model, tok, device, max_length=384):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import lib_cf as L
    out = {}
    model.eval()
    for name, (split_path, pages_dir) in EVAL_SPLITS.items():
        ev = json.load(open(split_path, encoding="utf-8"))
        for split in ("dev",):
            hits, mrrs = [], []
            for r in ev[split]:
                pages = L.load_pages(pages_dir, r["doc_id"])
                corpus = L.corpus_pages(pages)
                if not corpus:
                    continue
                gold, src = L.gold_pages_by_answer(pages, r["answer"], r["evidence_pages"])
                if src != "answer-string" and name == "mm":
                    continue
                bm, tf, rrf = L.retrieve(pages, r["question"])
                cand = rrf[:20]
                q = f"Question: {r['question']}\nAnswer: [NO_ANSWER]"
                scores = []
                B = 32
                for i in range(0, len(cand), B):
                    batch = tok([q] * len(cand[i:i + B]), [pages[p][:3000] for p in cand[i:i + B]],
                                truncation=True, max_length=max_length, padding=True,
                                return_tensors="pt").to(device)
                    with torch.no_grad():
                        s = model(**batch).logits.squeeze(-1)
                    scores.extend(s.tolist())
                order = [cand[i] for i in sorted(range(len(cand)), key=lambda j: -scores[j])]
                h1 = int(any(p in gold for p in order[:1]))
                mrr = next((1.0 / (i + 1) for i, p in enumerate(order) if p in gold), 0.0)
                hits.append(h1)
                mrrs.append(mrr)
            out[f"{name}_{split}"] = {"hit1": round(sum(hits) / max(len(hits), 1), 3),
                                      "mrr": round(sum(mrrs) / max(len(mrrs), 1), 3),
                                      "n": len(hits)}
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/raw/mixed_train_pairs.jsonl")
    ap.add_argument("--gen", default="results/raw/gen_answers_train.jsonl")
    ap.add_argument("--manifest", default="results/raw/gen_manifest_train.jsonl")
    ap.add_argument("--out", default="models/formal_v1")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=2024201433)
    ap.add_argument("--source-filter", default="",
                    help="train on one source only, e.g. qasper (empty = all/mixed)")
    ap.add_argument("--max-questions", type=int, default=0,
                    help="smoke-test limit; 0 means all questions")
    ap.add_argument("--skip-eval", action="store_true",
                    help="skip dev evaluation; intended only for smoke tests")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(args.device)

    rows, slots, n_questions = load_pairs(
        args.pairs, args.gen, args.manifest, args.source_filter, args.max_questions)
    random.shuffle(rows)
    print(f"questions={n_questions} pairs={len(rows)} by_slot={dict(slots)}", flush=True)

    pos = sum(1 for r in rows if r["label"] == 1)
    neg = len(rows) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)]).to(args.device)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Pre-tokenize once with the fast (Rust) tokenizer in big batches; serving ready
    # tensors removes the single-threaded __getitem__ bottleneck entirely.
    print("pre-tokenizing...", flush=True)
    enc = tok([r["query"] for r in rows], [r["page_text"] for r in rows],
              truncation=True, max_length=args.max_length, padding="max_length",
              return_tensors="pt")
    labels = torch.tensor([float(r["label"]) for r in rows])
    ds = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"], labels)
    del enc
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, pin_memory=(args.device == "cuda"))
    print(f"tensorized: {len(ds)} items, {len(dl)} steps/epoch", flush=True)

    history, best = [], None
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for bi, (ids, mask, y) in enumerate(dl, 1):
            batch = {"input_ids": ids.to(args.device), "attention_mask": mask.to(args.device)}
            y = y.to(args.device)
            opt.zero_grad()
            logits = model(**batch).logits.squeeze(-1)
            loss = lossf(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
            if bi % 50 == 0:
                print(f"ep{ep} step{bi}/{len(dl)} loss={tot/nb:.4f}", flush=True)
        devm = {} if args.skip_eval else evaluate(model, tok, args.device, args.max_length)
        history.append({"epoch": ep, "loss": round(tot / max(nb, 1), 6), "dev": devm})
        print(f"== epoch {ep}: loss={tot/max(nb,1):.4f} dev={devm}", flush=True)
        torch.save(model.state_dict(), os.path.join(args.out, f"epoch{ep}.pt"))
    json.dump({"base": MODEL, "args": vars(args), "slots": dict(slots), "history": history,
               "questions": n_questions,
               "note": "formal run; input candidates deduplicated by page/label; eval slot = none"},
              open(os.path.join(args.out, "train_log.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
