#!/usr/bin/env python
"""S6: fill the 'gen' answer slot — qwen-plus answers on train-pool questions (gold evidence in
context, so answers include both correct and wrong ones; used only from train split).

Env: OPENAI_API_KEY (required), OPENAI_BASE_URL, OPENAI_MODEL (default qwen-plus).
Caches per qid; resumable.

Usage: python -I s6_gen_train_answers.py <gen_manifest.jsonl> <out_jsonl>
"""
import json
import os
import sys
import time

import httpx

SYSTEM = (
    "You are a careful document QA assistant. Read the provided document pages and answer the "
    'question. Reply with ONLY a JSON object: {"answer": "<short answer>"}; if the pages do not '
    'contain the answer, reply {"answer": "insufficient information"}.'
)


def chat(base, key, model, user_text, retries=3):
    payload = {"model": model, "messages": [{"role": "system", "content": SYSTEM},
                                             {"role": "user", "content": user_text}],
               "temperature": 0, "max_tokens": 120}
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for a in range(retries):
        try:
            r = httpx.post(f"{base.rstrip('/')}/chat/completions", json=payload, headers=hdr,
                           timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"], r.json().get("usage", {})
        except Exception as e:
            if a == retries - 1:
                return None, {"error": repr(e)}
            time.sleep(3 * (a + 1))


def main(manifest_path, out_path):
    key = os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.environ.get("OPENAI_MODEL", "qwen-plus")
    if not key:
        print("ERROR: OPENAI_API_KEY not set; nothing fabricated.")
        sys.exit(2)
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            d = json.loads(line)
            done[d["qid"]] = d
    tin = tout = 0
    for i, line in enumerate(open(manifest_path, encoding="utf-8"), 1):
        m = json.loads(line)
        if m["qid"] in done:
            continue
        user = m["context_text"] + f"\n\nQuestion: {m['question']}"
        raw, usage = chat(base, key, model, user)
        try:
            ans = json.loads(raw).get("answer", "")
        except Exception:
            ans = raw or ""
        d = {"qid": m["qid"], "gen_answer": ans,
             "matches_gold": ans.strip().lower() == m["gold_answer"].strip().lower()}
        tin += (usage or {}).get("prompt_tokens", 0) or 0
        tout += (usage or {}).get("completion_tokens", 0) or 0
        done[m["qid"]] = d
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        if i % 50 == 0:
            print(f"[{i}] in={tin} out={tout}", flush=True)
    n = len(done)
    print(json.dumps({"answered": n, "match_gold_rate": round(
        sum(1 for d in done.values() if d.get("matches_gold")) / max(n, 1), 3),
        "prompt_tokens": tin, "completion_tokens": tout}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
