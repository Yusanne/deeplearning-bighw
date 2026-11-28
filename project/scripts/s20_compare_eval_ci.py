#!/usr/bin/env python
"""Paired bootstrap comparison between two S17 evaluation outputs."""
import json
import random
import sys


def mean(xs):
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


def ci(values, rng, n_boot):
    observed = mean(values)
    boots = [mean(values[rng.randrange(len(values))] for _ in values)
             for _ in range(n_boot)]
    boots.sort()
    return {"value": round(observed, 4),
            "ci95": [round(boots[int(0.025 * len(boots))], 4),
                     round(boots[min(int(0.975 * len(boots)), len(boots) - 1)], 4)]}


def main():
    if len(sys.argv) < 4:
        raise SystemExit("usage: s20_compare_eval_ci.py A.json B.json OUT.json [N_BOOT] [METHOD]")
    a_path, b_path, out_path = sys.argv[1:4]
    n_boot = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    method = sys.argv[5] if len(sys.argv) > 5 else "finetuned_gen"
    a = json.load(open(a_path, encoding="utf-8"))
    b = json.load(open(b_path, encoding="utf-8"))
    a_rows = {(r["split"], r["qid"]): r for r in a.get("rows", [])}
    b_rows = {(r["split"], r["qid"]): r for r in b.get("rows", [])}
    keys = sorted(set(a_rows) & set(b_rows))
    rng = random.Random(2024201433)
    result = {"a": a_path, "b": b_path, "method": method,
              "n_boot": n_boot, "splits": {}}
    for split in ("dev", "test"):
        split_keys = [k for k in keys if k[0] == split]
        if not split_keys:
            continue
        out = {"n": len(split_keys), "delta_b_minus_a": {}}
        for i, metric in enumerate(("hit1", "hit5", "mrr")):
            deltas = [b_rows[k]["metrics"][method][i] - a_rows[k]["metrics"][method][i]
                      for k in split_keys]
            out["delta_b_minus_a"][metric] = ci(deltas, rng, n_boot)
        result["splits"][split] = out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
