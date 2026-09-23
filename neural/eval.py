#!/usr/bin/env python3
"""Score a trained model against the exact labels on the held-out split.

Usage:
    python neural/eval.py [--split test] [--checkpoint checkpoints/best.pt]

Average error is the obvious measure and the least interesting one. This model is
an approximation of a solver that is already exact, so the question is not whether
it is close on average but whether it is ever confidently wrong in the direction
that loses games:

* A cell the solver proves is a mine, which the model calls nearly safe, is a
  square a player following the model walks onto.
* A cell the solver proves is safe, which the model calls nearly certain death, is
  only a wasted opportunity — worth knowing, but it does not end a game.

So the summary below separates plain accuracy from those two, and reports the
worst case rather than the mean, because a mean over millions of cells hides
exactly the rare confident mistake that matters.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import MinesweeperDataset
from model import PatchCNN

DATA_DIR = Path(__file__).parent / "data"
CKPT_DEFAULT = Path(__file__).parent / "checkpoints" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", type=Path, default=CKPT_DEFAULT)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"no checkpoint at {args.checkpoint} — train first")

    dataset = MinesweeperDataset(DATA_DIR / args.split, augment=False)
    total = len(dataset) if not args.limit else min(args.limit, len(dataset))
    print(f"{args.split}: {total:,} cells")

    model = PatchCNN()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.eval()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(total)),
        batch_size=args.batch,
        num_workers=4,
    )

    predicted = np.empty(total, dtype=np.float32)
    actual = np.empty(total, dtype=np.float32)
    at = 0
    with torch.no_grad():
        for patches, labels in loader:
            out = model(patches).numpy()
            predicted[at : at + len(out)] = out
            actual[at : at + len(out)] = labels.numpy()
            at += len(out)

    error = np.abs(predicted - actual)
    print(f"  mean absolute error : {error.mean():.4f}")
    print(f"  median              : {np.median(error):.4f}")
    print(f"  99th percentile     : {np.percentile(error, 99):.4f}")
    print(f"  worst               : {error.max():.4f}")

    # The dangerous direction: proven mines the model would let a player open.
    certain_mine = actual > 0.999
    if certain_mine.any():
        called_safe = (predicted[certain_mine] < 0.10).mean()
        print(
            f"\n  cells the solver proves are mines : {certain_mine.sum():,}"
            f"\n    model says under 10%            : {called_safe:.3%}  <- these lose games"
        )

    # The wasteful direction: proven-safe cells the model would have you avoid.
    certain_safe = actual < 0.001
    if certain_safe.any():
        called_risky = (predicted[certain_safe] > 0.50).mean()
        print(
            f"  cells the solver proves are safe  : {certain_safe.sum():,}"
            f"\n    model says over 50%             : {called_risky:.3%}  <- only wasted moves"
        )

    # Calibration: among cells the model calls ~p, how many really are mines?
    print("\n  calibration")
    edges = [0.0, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 1.01]
    for low, high in zip(edges, edges[1:]):
        bucket = (predicted >= low) & (predicted < high)
        if bucket.sum() < 50:
            continue
        print(
            f"    model says {low:.2f}-{high:.2f}: "
            f"{bucket.sum():>9,} cells, actually {actual[bucket].mean():.3f}"
        )


if __name__ == "__main__":
    main()
