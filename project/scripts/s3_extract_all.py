#!/usr/bin/env python
"""S3: extract per-page text for every downloaded PDF into data/pages/*.json.

Usage: python -I s3_extract_docs.py <documents_dir> <pages_dir>
"""
import json
import os
import sys

import pymupdf


def main(docs_dir, pages_dir):
    ok, poor, fail = 0, 0, []
    for fn in sorted(os.listdir(docs_dir)):
        if not fn.lower().endswith(".pdf"):
            continue
        dest = os.path.join(pages_dir, fn + ".json")
        if os.path.exists(dest):
            continue
        try:
            doc = pymupdf.open(os.path.join(docs_dir, fn))
            pages, n_poor = {}, 0
            page_count = doc.page_count
            for i in range(page_count):
                t = doc[i].get_text() or ""
                pages[str(i)] = t
                if len(t.strip()) < 50:
                    n_poor += 1
            doc.close()
            json.dump(pages, open(dest, "w", encoding="utf-8"), ensure_ascii=False)
            ok += 1
            poor += n_poor
            if n_poor > page_count * 0.5:
                print(f"WARN mostly-poor: {fn} ({n_poor}/{page_count} pages thin)", flush=True)
        except Exception as e:
            fail.append((fn, repr(e)[:80]))
    print(f"extracted={ok} thin_pages_total={poor} fail={len(fail)}")
    if fail:
        print(json.dumps(fail, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
