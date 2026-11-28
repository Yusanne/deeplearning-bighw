#!/usr/bin/env python
"""Paired bootstrap confidence intervals for S17/S18 result JSON files.

The input must contain ``rows`` with a per-question ``metrics`` mapping.  The
reference method defaults to ``rrf`` for the trained reranker output and can
be set to ``rrf_dense`` for the Dense baseline output.
"""
import json
import random
import sys


def mean(xs):
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


def ci(values, rng, n_boot):
    if not values:
        return {"value": 0.0, "ci95": [0.0, 0.0]}
    observed = mean(values)
    boots = []
    for _ in range(n_boot):
        boots.append(mean(values[rng.randrange(len(values))] for _ in values))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[min(int(0.975 * len(boots)), len(boots) - 1)]
    return {"value": round(observed, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: s19_bootstrap_ci.py INPUT.json OUTPUT.json [N_BOOT] [REFERENCE]")
    in_path, out_path = sys.argv[1:3]
    n_boot = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    requested_ref = sys.argv[4] if len(sys.argv) > 4 else ""
    data = json.load(open(in_path, encoding="utf-8"))
    rng = random.Random(2024201433)
    result = {"source": in_path, "n_boot": n_boot, "splits": {}}
    metric_names = ("hit1", "hit5", "mrr")
    for split in ("dev", "test"):
        rows = [r for r in data.get("rows", []) if r.get("split") == split]
        if not rows:
            continue
        methods = list(rows[0].get("metrics", {}).keys())
        reference = requested_ref or ("rrf" if "rrf" in methods else "rrf_dense")
        if reference not in methods:
            raise ValueError(f"reference {reference!r} not in methods {methods}")
        vals = {m: {metric: [r["metrics"][m][i] for r in rows]
                    for i, metric in enumerate(metric_names)} for m in methods}
        split_out = {"n": len(rows), "reference": reference,
                     "absolute": {}, "delta_vs_reference": {}}
        for method in methods:
            split_out["absolute"][method] = {
                metric: ci(vals[method][metric], rng, n_boot)
                for metric in metric_names
            }
            if method != reference:
                split_out["delta_vs_reference"][method] = {}
                for metric in metric_names:
                    diffs = [a - b for a, b in zip(vals[method][metric], vals[reference][metric])]
                    split_out["delta_vs_reference"][method][metric] = ci(diffs, rng, n_boot)
        result["splits"][split] = split_out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
