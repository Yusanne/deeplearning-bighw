#!/usr/bin/env python
"""S6b: concurrent generation-answer runner for both manifests (qwen-plus, temperature 0).
Caches per qid; resumable; ~8 workers.

Env: OPENAI_API_KEY (required), OPENAI_BASE_URL, OPENAI_MODEL (default qwen-plus).
Usage: python -I s6b_gen_answers_concurrent.py <manifest.jsonl> <out.jsonl> [workers]
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

SYSTEM = ("You are a careful document QA assistant. Read the provided document pages and answer "
          'the question. Reply with ONLY a JSON object: {"answer": "<short answer>"}; if the '
          'pages do not contain the answer, reply {"answer": "insufficient information"}.')

_lock = threading.Lock()


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
            d = r.json()
            return d["choices"][0]["message"]["content"], d.get("usage", {})
        except Exception:
            if a == retries - 1:
                return None, {}
            time.sleep(2 * (a + 1))


def main(manifest_path, out_path, workers=8):
    key = os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.environ.get("OPENAI_MODEL", "qwen-plus")
    if not key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(2)
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            try:
                d = json.loads(line)
                if not d.get("failed") and d.get("gen_answer"):
                    done[d["qid"]] = d  # failed rows are retried on resume
            except Exception:
                pass
    todo = []
    for line in open(manifest_path, encoding="utf-8"):
        m = json.loads(line)
        if m["qid"] not in done:
            todo.append(m)
    print(f"total={len(done)+len(todo)} todo={len(todo)}", flush=True)
    stats = {"in": 0, "out": 0, "fail": 0}

    def work(m):
        raw, usage = chat(base, key, model,
                          m["context_text"] + f"\n\nQuestion: {m['question']}")
        try:
            ans = json.loads(raw).get("answer", "") if raw else ""
        except Exception:
            ans = raw or ""
        u = usage or {}
        d = {"qid": m["qid"], "gen_answer": ans,
             "prompt_tokens": u.get("prompt_tokens", 0) or 0,
             "completion_tokens": u.get("completion_tokens", 0) or 0,
             "failed": raw is None}
        with _lock:
            stats["in"] += d["prompt_tokens"]
            stats["out"] += d["completion_tokens"]
            stats["fail"] += int(d["failed"])
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, m) for m in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            fu.result()
            if i % 100 == 0:
                print(f"[{i}/{len(todo)}] in={stats['in']} out={stats['out']} "
                      f"fail={stats['fail']} {time.time()-t0:.0f}s", flush=True)
    print(json.dumps({"done_total": len(done) + len(todo), **stats,
                      "seconds": round(time.time() - t0)}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
