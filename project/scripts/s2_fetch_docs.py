#!/usr/bin/env python
"""S2: download all PDFs listed in the MMLongBench-Doc documents folder.

Skips files already present (>10KB). Resilient: logs failures, continues.

Usage: python -I s2_fetch_docs.py <doc_list.json> <documents_dir>
"""
import json
import os
import sys
import time

import requests


def main(doclist_path, outdir):
    listing = json.load(open(doclist_path, encoding="utf-8"))
    todo = [(e["name"], e.get("download_url")) for e in listing if e.get("type") == "file"]
    ok, skip, fail = [], [], []
    for i, (name, url) in enumerate(todo, 1):
        dest = os.path.join(outdir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 10000:
            skip.append(name)
            continue
        if not url:
            fail.append((name, "no-url"))
            continue
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=300)
                if r.status_code == 200 and r.content[:5] == b"%PDF-":
                    with open(dest, "wb") as f:
                        f.write(r.content)
                    ok.append(name)
                    print(f"[{i}/{len(todo)}] OK {name} {len(r.content)//1024}KB", flush=True)
                    break
                fail.append((name, f"http{r.status_code}"))
                break
            except Exception as e:
                if attempt == 2:
                    fail.append((name, repr(e)[:80]))
                else:
                    time.sleep(5 * (attempt + 1))
    print(f"done ok={len(ok)} skip={len(skip)} fail={len(fail)}")
    if fail:
        print("FAILED:", json.dumps(fail, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
