#!/usr/bin/env python
"""S8: probe and (optionally) download LongDocURL subset of MMLongBench; report availability.

MMLongBench distributes page *images* for docqa tasks (not PDFs); text must come from their
OCR ablation files if present. This script reports what is available and flags the gap.

Usage: python -I s8_longdocurl_check.py <outdir> [--download]
"""
import json
import os
import sys

import requests

HF_API = "https://huggingface.co/api/datasets/ZhaoweiWang/MMLongBench"
CANDIDATE_REPO = "ZhaoweiWang/MMLongBench"
# NOTE: docqa pages are IMAGES (page_list -> *.jpg); no OCR text ships with the dataset.
# External-validation pipeline: 0_mmlb_data.tar.gz (jsonl) + 5_docqa_image.tar.gz (jpg)
# -> s10_ocr_longdocurl.py (qwen-vl-ocr via DashScope) -> pages json compatible with lib_cf.


def main(outdir, download=False):
    os.makedirs(outdir, exist_ok=True)
    # The API is occasionally unavailable from campus networks.  Keep the
    # official repo/task facts in the dry-run output instead of failing before
    # the user can see that the archive is intentionally too large.
    try:
        resp = requests.get(HF_API, timeout=30)
        info = resp.json() if resp.ok else {}
    except Exception:
        info = {}
    files = [s["rfilename"] for s in info.get("siblings", [])]
    longdoc = [f for f in files if "longdocurl" in f.lower() or f.endswith(".tar.gz")]
    archive = "0_mmlb_data.tar.gz"
    archive_url = f"https://huggingface.co/datasets/{CANDIDATE_REPO}/resolve/main/{archive}"
    size = None
    try:
        h = requests.head(archive_url, allow_redirects=True, timeout=30)
        size = int(h.headers.get("content-length", "0")) or None
    except Exception:
        pass
    print(json.dumps({"repo": CANDIDATE_REPO, "n_files": len(files),
                      "longdocurl_or_archive": longdoc[:20],
                      "known_archive": archive, "archive_size_bytes": size,
                      "note": "LongDocURL is an MMLongBench task; default project validation uses QASPER",
                      "downloaded": False}, ensure_ascii=False, indent=2))
    if not download:
        print("dry-run only; do not download the ~9.98GB visual archive for the main project")
        return
    raise SystemExit("refusing large LongDocURL archive in the default workflow; use QASPER external validation")
    for f in longdoc:
        if not (f.endswith(".jsonl") or f.endswith(".tar.gz")):
            continue
        url = f"https://huggingface.co/datasets/{CANDIDATE_REPO}/resolve/main/{f}"
        dest = os.path.join(outdir, os.path.basename(f))
        r = requests.get(url, timeout=600)
        if r.status_code == 200:
            open(dest, "wb").write(r.content)
            print("saved", dest, len(r.content) // 1024, "KB")
        else:
            print("skip", f, r.status_code)
    for f in sorted(os.listdir(outdir)):
        if f.endswith(".jsonl"):
            n = sum(1 for _ in open(os.path.join(outdir, f), encoding="utf-8"))
            first = json.loads(open(os.path.join(outdir, f), encoding="utf-8").readline())
            print(f, "rows=", n, "fields=", list(first.keys()))


if __name__ == "__main__":
    main(sys.argv[1], "--download" in sys.argv)
