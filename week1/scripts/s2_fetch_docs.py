#!/usr/bin/env python
"""Week1 S2: download only the PDFs needed by the selected questions.

Usage: python -I s2_fetch_docs.py <selected20.json> <doc_list.json> <documents_dir>
"""
import json
import os
import sys

import requests


def main(selected_path, doclist_path, outdir):
    sel = json.load(open(selected_path, encoding="utf-8"))
    docs = sorted({q["doc_id"] for q in sel["questions"]})
    listing = json.load(open(doclist_path, encoding="utf-8"))
    by_name = {e["name"]: e.get("download_url") for e in listing}
    ok, fail = [], []
    for d in docs:
        dest = os.path.join(outdir, d)
        if os.path.exists(dest) and os.path.getsize(dest) > 10000:
            ok.append(d)
            continue
        url = by_name.get(d)
        if not url:
            fail.append((d, "no-url"))
            continue
        try:
            r = requests.get(url, timeout=240)
            if r.status_code == 200 and r.content[:5] == b"%PDF-":
                with open(dest, "wb") as f:
                    f.write(r.content)
                ok.append(d)
                print("OK", d, len(r.content) // 1024, "KB")
            else:
                fail.append((d, f"http {r.status_code}"))
                print("FAIL", d, r.status_code)
        except Exception as e:
            fail.append((d, repr(e)))
            print("FAIL", d, repr(e))
    print(f"done: {len(ok)} ok / {len(docs)} needed; fail={fail}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
