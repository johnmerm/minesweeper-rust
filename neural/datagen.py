#!/usr/bin/env python3
"""Spawn N parallel datagen processes, merge JSONL output, split train/val/test.

Usage:
    python neural/datagen.py 500000 8
    # Generates 500k samples using 8 parallel processes.
    # Output files: neural/data/{train,val,test}.jsonl
"""

import argparse
import math
import os
import random
import subprocess
import sys
import threading
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATAGEN_BIN = Path(__file__).parent.parent / "target" / "release" / "datagen"

SPLIT_FRACS = {"train": 0.85, "val": 0.10, "test": 0.05}


def run_worker(n: int, out_file, lock: threading.Lock, pbar_state: dict):
    """Run one datagen process producing `n` samples, writing lines to out_file."""
    proc = subprocess.Popen(
        [str(DATAGEN_BIN), str(n)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    written = 0
    for line in proc.stdout:
        line = line.rstrip("\n")
        if not line:
            continue
        with lock:
            out_file.write(line + "\n")
            out_file.flush()
            written += 1
            pbar_state["done"] += 1
            total = pbar_state["total"]
            done = pbar_state["done"]
            pct = done / total * 100
            print(f"\r  {done:>7}/{total}  ({pct:.1f}%)  ", end="", flush=True)
    proc.wait()


def split_file(all_path: Path):
    """Shuffle and split the raw JSONL into train/val/test files."""
    print("\nShuffling & splitting...", flush=True)
    with open(all_path) as f:
        lines = f.readlines()
    random.shuffle(lines)

    n = len(lines)
    n_train = int(n * SPLIT_FRACS["train"])
    n_val = int(n * SPLIT_FRACS["val"])

    splits = {
        "train": lines[:n_train],
        "val": lines[n_train : n_train + n_val],
        "test": lines[n_train + n_val :],
    }
    for name, chunk in splits.items():
        out = DATA_DIR / f"{name}.jsonl"
        with open(out, "w") as f:
            f.writelines(chunk)
        print(f"  {name:5}: {len(chunk):>7} samples → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_samples", type=int, default=500_000, nargs="?")
    parser.add_argument("n_procs", type=int, default=4, nargs="?")
    args = parser.parse_args()

    if not DATAGEN_BIN.exists():
        print(
            f"ERROR: datagen binary not found at {DATAGEN_BIN}\n"
            "Build it first: cargo build --release -p datagen",
            file=sys.stderr,
        )
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_path = DATA_DIR / "all.jsonl"

    n_per_proc = math.ceil(args.n_samples / args.n_procs)
    print(
        f"Generating {args.n_samples} samples with {args.n_procs} workers "
        f"({n_per_proc} each)..."
    )

    lock = threading.Lock()
    pbar_state = {"done": 0, "total": args.n_samples}

    with open(all_path, "w") as out_file:
        threads = []
        for _ in range(args.n_procs):
            t = threading.Thread(
                target=run_worker, args=(n_per_proc, out_file, lock, pbar_state)
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    print()  # newline after progress
    split_file(all_path)
    all_path.unlink()  # remove the merged temp file
    print("Done.")


if __name__ == "__main__":
    main()
