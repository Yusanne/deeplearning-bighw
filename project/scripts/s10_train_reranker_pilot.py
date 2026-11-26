#!/usr/bin/env python
"""S10: train a small pilot cross-encoder on the actually available pairs.

This is deliberately a feasibility run, not the final A+ training recipe.  It
uses the document-isolated pairs emitted by S5 and records the exact number of
questions/pairs, so a small corpus cannot be accidentally reported as a large
supervised dataset.

Usage:
  python -I s10_train_reranker_pilot.py <train_pairs.jsonl> <out_model_dir> [epochs]
"""
import json
import os
import sys


def main(pairs_path, out_dir, epochs=2):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
    max_rows = int(os.environ.get("MAX_ROWS", "0"))
    rows = [json.loads(x) for x in open(pairs_path, encoding="utf-8") if x.strip()]
    if max_rows:
        rows = rows[:max_rows]
    if not rows:
        raise SystemExit("no training pairs")

    torch.manual_seed(2024201433)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    net = AutoModelForSequenceClassification.from_pretrained(model_name)
    net.to(device)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-5, weight_decay=0.01)
    pos_n = sum(int(r["label"]) for r in rows)
    neg_n = len(rows) - pos_n
    pos_weight = float(neg_n / max(pos_n, 1)) if os.environ.get("BALANCE_CLASSES", "1") != "0" else 1.0
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))
    history = []

    batch_size = int(os.environ.get("BATCH_SIZE", "32" if device.type == "cuda" else "8"))
    max_length = int(os.environ.get("MAX_LENGTH", "512"))
    # Keep the length in one place so the formal run and its metadata agree.
    def collate(batch):
        left = [f"Question: {r['question']}\nAnswer: {r['answer_text']}" for r in batch]
        right = [r["page_text"] for r in batch]
        enc = tok(left, right, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        y = torch.tensor([float(r["label"]) for r in batch], device=device)
        return enc, y

    loader = DataLoader(rows, batch_size=batch_size, shuffle=True, collate_fn=collate)

    for ep in range(1, int(epochs) + 1):
        total = 0.0
        n = 0
        for enc, y in loader:
            out = net(**enc).logits.squeeze(-1)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            total += float(loss.item()) * len(y)
            n += len(y)
        rec = {"epoch": ep, "loss": round(total / max(n, 1), 6)}
        history.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    os.makedirs(out_dir, exist_ok=True)
    net.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    meta = {
        "base_model": model_name,
        "pairs": len(rows),
        "questions": len({r["qid"] for r in rows}),
        "device": str(device),
        "batch_size": batch_size,
        "positive": pos_n,
        "negative": neg_n,
        "pos_weight": round(pos_weight, 6),
        "epochs": int(epochs),
        "max_length": max_length,
        "learning_rate": 2e-5,
        "history": history,
        "note": "pilot only; evaluation remains document-isolated and uses generated/no-answer slots",
    }
    json.dump(meta, open(os.path.join(out_dir, "pilot_train.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2)
