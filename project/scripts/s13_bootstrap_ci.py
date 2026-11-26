#!/usr/bin/env python
"""Paired bootstrap confidence intervals for S7 per-question metrics."""
import json
import random
import sys


def mean(xs):
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


def main(in_path, out_path, n_boot=2000):
    d = json.load(open(in_path, encoding="utf-8"))
    rng = random.Random(2024201433)
    result = {"source": in_path, "n_boot": int(n_boot), "splits": {}}
    methods = ["bm25", "tfidf", "rrf", "zs_oracle", "zs_none"]
    for split in ("dev", "test"):
        rows = [r for r in d.get("rows", []) if r.get("split") == split]
        if not rows:
            continue
        split_out = {"n": len(rows), "absolute": {}, "delta_vs_rrf": {}}
        for metric_idx, metric_name in enumerate(("hit1", "hit5", "mrr")):
            vals = {m: [r["metrics"][m][metric_idx] for r in rows] for m in methods}
            for m in methods:
                observed = mean(vals[m])
                boots = []
                for _ in range(int(n_boot)):
                    sample = [vals[m][rng.randrange(len(rows))] for _ in rows]
                    boots.append(mean(sample))
                boots.sort()
                split_out["absolute"].setdefault(m, {})[metric_name] = {
                    "value": round(observed, 4),
                    "ci95": [round(boots[int(0.025 * len(boots))], 4),
                             round(boots[int(0.975 * len(boots))], 4)]
                }
            for m in methods:
                if m == "rrf":
                    continue
                observed = mean([a - b for a, b in zip(vals[m], vals["rrf"])])
                boots = []
                for _ in range(int(n_boot)):
                    idx = [rng.randrange(len(rows)) for _ in rows]
                    boots.append(mean(vals[m][i] - vals["rrf"][i] for i in idx))
                boots.sort()
                split_out["delta_vs_rrf"].setdefault(m, {})[metric_name] = {
                    "value": round(observed, 4),
                    "ci95": [round(boots[int(0.025 * len(boots))], 4),
                             round(boots[int(0.975 * len(boots))], 4)]
                }
        result["splits"][split] = split_out
    json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2000)
