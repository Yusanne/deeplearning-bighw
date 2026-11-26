#!/usr/bin/env python
"""S10: OCR LongDocURL page images into lib_cf-compatible pages json (qwen-vl-ocr via DashScope).

Selects the eval questions' docs from longdocurl jsonl, OCRs each page image once (cached),
writes data/pages_longdocurl/<doc>.json. Only run after 5_docqa_image.tar.gz is unpacked.

Env: OPENAI_API_KEY, OPENAI_BASE_URL (default dashscope compatible), OCR_MODEL (qwen-vl-ocr).
Usage: python -I s10_ocr_longdocurl.py <longdocurl.jsonl> <images_dir> <out_pages_dir> <cache_dir>
"""
import base64
import json
import os
import sys
import time

import httpx
import requests


def select_docs(jsonl_path, max_docs=20):
    """Pick questions from the fewest distinct docs to cap OCR volume."""
    rows = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]
    rows = [r for r in rows if str(r.get("answer_format", "")).lower() not in ("none", "null")
            and str(r.get("answer", "")).strip()]
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r["doc_name"], []).append(r)
    docs = sorted(by_doc, key=lambda d: len(by_doc[d]), reverse=True)[:max_docs]
    picked = []
    for d in docs:
        picked.extend(by_doc[d][:3])  # cap 3 questions/doc
        if len(picked) >= 40:
            break
    return picked


def ocr_page(base, key, model, img_path, retries=3):
    b64 = base64.b64encode(open(img_path, "rb").read()).decode()
    payload = {"model": model, "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": "Extract all text from this document page. Output plain text only."},
        ]}]}
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for a in range(retries):
        try:
            r = httpx.post(f"{base.rstrip('/')}/chat/completions", json=payload, headers=hdr,
                           timeout=180)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except Exception:
            if a == retries - 1:
                return ""
            time.sleep(3 * (a + 1))


def main(jsonl_path, images_dir, out_dir, cache_dir):
    key = os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.environ.get("OCR_MODEL", "qwen-vl-ocr")
    if not key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(2)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    picked = select_docs(jsonl_path)
    docs = sorted({r["doc_name"] for r in picked})
    print(f"questions={len(picked)} docs={len(docs)}")
    json.dump(picked, open(os.path.join(out_dir, "_selected40.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    n_done = n_missing = 0
    for d in docs:
        dest = os.path.join(out_dir, f"{d}.json")
        if os.path.exists(dest):
            n_done += 1
            continue
        pages = {}
        i = 0
        while True:
            img = os.path.join(images_dir, "longdocurl", d, f"{d}_page{i}.jpg")
            if not os.path.exists(img):
                break
            cache = os.path.join(cache_dir, f"{d}_page{i}.txt")
            if os.path.exists(cache):
                txt = open(cache, encoding="utf-8").read()
            else:
                txt = ocr_page(base, key, model, img)
                open(cache, "w", encoding="utf-8").write(txt)
            pages[str(i)] = txt
            i += 1
        if not pages:
            n_missing += 1
            continue
        json.dump(pages, open(dest, "w", encoding="utf-8"), ensure_ascii=False)
        n_done += 1
        print(f"doc {d}: {i} pages OCR'd", flush=True)
    print(f"done docs={n_done} missing_images_for={n_missing}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
