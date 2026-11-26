#!/usr/bin/env python
"""Week1 S5: run four-world generation through an OpenAI-compatible API, cache responses,
compute behavior-change flags and cost. Requires API key; nothing is fabricated without it.

Env:
  OPENAI_API_KEY   (required)
  OPENAI_BASE_URL  (default https://api.deepseek.com)
  OPENAI_MODEL     (default deepseek-chat)

Usage: python -I s5_run_llm.py <worlds_dir> <llm_out_dir>
"""
import json
import os
import re
import sys
import time

import httpx

SYSTEM = (
    "You are a careful document QA assistant. Read the provided document pages and "
    'answer the question. Reply with ONLY a JSON object: {"answer": "<short answer>", '
    '"pages": [<page numbers you actually used>]}. If the pages do not contain the '
    'answer, reply {"answer": "insufficient information", "pages": []}.'
)


def norm(s):
    return re.sub(r"[%$€£'\"\s(),.]+", "", str(s)).lower()


def build_user(row, world):
    parts = [f"[Page {c['page']}]\n{c['text']}" for c in row["worlds"][world]]
    return "\n\n".join(parts) + f"\n\nQuestion: {row['question']}"


def chat(base, key, model, messages, max_retries=2):
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 200}
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for a in range(max_retries + 1):
        try:
            r = httpx.post(f"{base.rstrip('/')}/chat/completions", json=payload,
                           headers=hdr, timeout=120)
            r.raise_for_status()
            d = r.json()
            return d["choices"][0]["message"]["content"], d.get("usage", {})
        except Exception as e:
            if a == max_retries:
                return None, {"error": repr(e)}
            time.sleep(3 * (a + 1))


def main(worlds_dir, out_dir):
    key = os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("OPENAI_MODEL", "deepseek-chat")
    if not key:
        print("ERROR: set OPENAI_API_KEY (and optional OPENAI_BASE_URL / OPENAI_MODEL). "
              "No API call was made; nothing is fabricated.")
        sys.exit(2)
    os.makedirs(out_dir, exist_ok=True)
    total_in = total_out = 0
    cost_rows = []
    files = sorted(f for f in os.listdir(worlds_dir) if f.endswith(".json"))
    for fn in files:
        row = json.load(open(os.path.join(worlds_dir, fn), encoding="utf-8"))
        answers = {}
        for world in ("plus", "del", "swap", "benign"):
            cache = os.path.join(out_dir, f"{row['qid']}__{world}.json")
            if os.path.exists(cache):
                d = json.load(open(cache, encoding="utf-8"))
            else:
                txt, usage = chat(base, key, model,
                                  [{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": build_user(row, world)}])
                d = {"raw": txt, "usage": usage}
                json.dump(d, open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            answers[world] = d
            u = d.get("usage", {}) or {}
            total_in += u.get("prompt_tokens", 0) or 0
            total_out += u.get("completion_tokens", 0) or 0
        try:
            a_plus = json.loads(answers["plus"]["raw"]).get("answer", "")
        except Exception:
            a_plus = answers["plus"]["raw"] or ""
        flags = {}
        for world in ("del", "swap", "benign"):
            try:
                aw = json.loads(answers[world]["raw"]).get("answer", "")
            except Exception:
                aw = answers[world]["raw"] or ""
            flags[world] = {
                "answer": aw,
                "same_as_plus": bool(a_plus) and norm(aw) == norm(a_plus),
                "says_insufficient": "insufficient" in (aw or "").lower(),
            }
        cost_rows.append({"qid": row["qid"], "a_plus": a_plus, "flags": flags,
                          "swap_valid_proxy": row["swap"]["valid_proxy"]})
    n = len(cost_rows)
    summary = {
        "questions": n,
        "total_prompt_tokens": total_in,
        "total_completion_tokens": total_out,
        "del_world_persists_answer": round(sum(r["flags"]["del"]["same_as_plus"] for r in cost_rows) / max(n, 1), 3),
        "swap_world_persists_answer": round(sum(r["flags"]["swap"]["same_as_plus"] for r in cost_rows) / max(n, 1), 3),
        "benign_world_persists_answer": round(sum(r["flags"]["benign"]["same_as_plus"] for r in cost_rows) / max(n, 1), 3),
    }
    json.dump({"summary": summary, "rows": cost_rows},
              open(os.path.join(out_dir, "week1_llm_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
