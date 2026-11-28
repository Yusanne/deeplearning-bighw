#!/usr/bin/env python
"""ModelArts training entrypoint (thin wrapper around s16_train_formal.py).

ModelArts conventions (same as hw8): code+data arrive via OBS mount or git clone;
outputs must be written to /home/ma-user/work (auto-synced to OBS output).

Usage inside ModelArts notebook/job:
  python train_modelarts_entry.py --pairs data/qasper_train_pairs.jsonl \
      --gen data/gen_answers_train.jsonl --manifest data/gen_manifest_train.jsonl \
      --epochs 2 --device cuda
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--gen", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--source-filter", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    work = "/home/ma-user/work"
    out_local = os.path.join(work, "formal_qasper_only")
    os.makedirs(out_local, exist_ok=True)

    cmd = [sys.executable, "-u", os.path.join(HERE, "s16_train_formal.py"),
           "--pairs", args.pairs, "--out", out_local,
           "--epochs", str(args.epochs), "--max-length", str(args.max_length),
           "--batch-size", str(args.batch_size), "--lr", str(args.lr),
           "--device", "cuda"]
    if args.gen:
        cmd += ["--gen", args.gen]
    if args.manifest:
        cmd += ["--manifest", args.manifest]
    if args.source_filter:
        cmd += ["--source-filter", args.source_filter]
    if args.resume:
        cmd += ["--resume"]
    print("RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    # copy artifacts to ModelArts output dir
    for f in os.listdir(out_local):
        shutil.copy2(os.path.join(out_local, f), os.path.join(work, f))
    print("artifacts copied to", work, flush=True)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
