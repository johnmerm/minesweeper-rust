#!/usr/bin/env python3
"""Convert the JSONL from datagen into compact arrays the trainer can stream.

Usage:
    python neural/prepare.py            # converts data/{train,val,test}.jsonl
    python neural/prepare.py --split val

# Why this step exists

The trainer reads one 9x9 patch per hidden cell, so each game record expands into
about 64 training samples. Reading those straight from JSONL means holding every
parsed record in memory: a record costs ~22 KB as Python dicts, so the 500 000
samples the datagen README suggests need **11.5 GB** before `DataLoader` forks it
once per worker. That is what made training impossible rather than merely slow.

The same record is 768 bytes as three uint8 planes — 30x smaller — and the planes
are all the patch extractor actually reads. This script does that conversion once,
so the dataset becomes a memory-mapped array the operating system can page in and
out at will, and 500 000 records cost about 380 MB of disk and almost no RAM.

It also precomputes the border plane, which the old dataset recomputed for every
cell of every record — 64 times more often than necessary.

# Output, per split

    <split>_boards.npy   uint8  (records, 3, H, W)   state / content / border
    <split>_cells.npy    int32  (samples, 3)         record index, x, y
    <split>_labels.npy   f32    (samples,)           P(mine) from ConstraintSearch
    <split>_ratios.npy   f32    (records,)           mines remaining / hidden cells

Boards are padded to the largest in the split; `state` is set to the out-of-bounds
code outside each board's real extent, so padding is indistinguishable from the
edge of the world and needs no separate size table.
"""

import argparse
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"

# State codes as emitted by the Rust datagen.
STATE_HIDDEN = 0
STATE_VISIBLE = 1
STATE_FLAGGED = 2
# Ours, for cells outside the board.
STATE_OUT_OF_BOUNDS = 3

CONTENT_MINE = 9
CONTENT_NONE = 255


def border_plane(state: np.ndarray, content: np.ndarray) -> np.ndarray:
    """Unopened cells touching a visible number.

    Computed by shifting the mask of visible numbers in all eight directions,
    which is the whole board at once rather than a loop per cell.
    """
    numbered = (state == STATE_VISIBLE) & (content > 0) & (content < CONTENT_MINE)
    touching = np.zeros_like(numbered)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(numbered, dy, axis=0), dx, axis=1)
            # np.roll wraps; blank the rows and columns that came round the edge.
            if dy > 0:
                shifted[:dy, :] = False
            elif dy < 0:
                shifted[dy:, :] = False
            if dx > 0:
                shifted[:, :dx] = False
            elif dx < 0:
                shifted[:, dx:] = False
            touching |= shifted
    unopened = (state == STATE_HIDDEN) | (state == STATE_FLAGGED)
    return (touching & unopened).astype(np.uint8)


def convert(jsonl: Path, out_prefix: Path) -> None:
    records = []
    cells = []
    labels = []
    ratios = []

    with open(jsonl) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            height, width = record["height"], record["width"]
            grid = record["grid"]

            state = np.array(
                [[grid[y][x]["state"] for x in range(width)] for y in range(height)],
                dtype=np.uint8,
            )
            content = np.array(
                [[grid[y][x]["content"] for x in range(width)] for y in range(height)],
                dtype=np.uint8,
            )

            index = len(records)
            records.append(np.stack([state, content, border_plane(state, content)]))

            hidden = state == STATE_HIDDEN
            mines_left = record["mines_count"] - int((content == CONTENT_MINE).sum())
            ratios.append(mines_left / max(int(hidden.sum()), 1))

            probs = record["probs"]
            for y in range(height):
                for x in range(width):
                    if hidden[y, x]:
                        cells.append((index, x, y))
                        labels.append(probs[y][x])

    if not records:
        raise SystemExit(f"{jsonl} held no records")

    height = max(board.shape[1] for board in records)
    width = max(board.shape[2] for board in records)

    boards = np.empty((len(records), 3, height, width), dtype=np.uint8)
    boards[:, 0] = STATE_OUT_OF_BOUNDS
    boards[:, 1] = CONTENT_NONE
    boards[:, 2] = 0
    for i, board in enumerate(records):
        boards[i, :, : board.shape[1], : board.shape[2]] = board

    np.save(f"{out_prefix}_boards.npy", boards)
    np.save(f"{out_prefix}_cells.npy", np.array(cells, dtype=np.int32))
    np.save(f"{out_prefix}_labels.npy", np.array(labels, dtype=np.float32))
    np.save(f"{out_prefix}_ratios.npy", np.array(ratios, dtype=np.float32))

    megabytes = boards.nbytes / 1e6
    print(
        f"  {jsonl.name}: {len(records):,} records -> {len(cells):,} samples, "
        f"boards {megabytes:.1f} MB"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="append", help="defaults to train, val and test")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    splits = args.split or ["train", "val", "test"]
    for split in splits:
        jsonl = args.data_dir / f"{split}.jsonl"
        if not jsonl.exists():
            print(f"  {split}.jsonl missing, skipping")
            continue
        convert(jsonl, args.data_dir / split)


if __name__ == "__main__":
    main()
