#!/usr/bin/env python3
"""Check that the prepared arrays say the same thing as the JSONL they came from.

Usage:
    python neural/check.py [--samples 4000] [--split train]

Needs numpy only — no torch — so it can run anywhere the data is generated.

`prepare.py` and `dataset.py` between them rewrote how a patch is built, from a
Python loop over the raw records to a slice of preprocessed planes. The two must
agree exactly, and nothing else in the pipeline would notice if they stopped:
a wrong channel produces a model that trains happily and predicts badly. So this
rebuilds patches straight from the JSONL, following the channel table in
dataset.py's docstring literally, and compares.
"""

import argparse
import json
from pathlib import Path

import numpy as np

HALF = 4
PATCH = 2 * HALF + 1
N_CHANNELS = 8
STATE_HIDDEN, STATE_VISIBLE, STATE_FLAGGED = 0, 1, 2
CONTENT_MINE = 9


def reference_patch(state, content, border, cx, cy, ratio):
    """A patch built the slow, obvious way, from the channel table in dataset.py."""
    height, width = state.shape
    patch = np.zeros((N_CHANNELS, PATCH, PATCH), dtype=np.float32)
    for pi in range(PATCH):
        for pj in range(PATCH):
            gy, gx = cy + pi - HALF, cx + pj - HALF
            if gy < 0 or gy >= height or gx < 0 or gx >= width:
                patch[4, pi, pj] = 1.0
                continue
            s, c = state[gy, gx], content[gy, gx]
            if s == STATE_VISIBLE:
                patch[1, pi, pj] = 1.0
                if c < CONTENT_MINE:
                    patch[0, pi, pj] = c / 8.0
            elif s == STATE_HIDDEN:
                patch[2, pi, pj] = 1.0
            elif s == STATE_FLAGGED:
                patch[3, pi, pj] = 1.0
            if border[gy, gx]:
                patch[7, pi, pj] = 1.0
    patch[5, HALF, HALF] = 1.0
    patch[6, :, :] = ratio
    return patch


def reference_border(state, content):
    """Unopened cells next to a visible number, by direct neighbour inspection."""
    height, width = state.shape
    border = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            if state[y, x] not in (STATE_HIDDEN, STATE_FLAGGED):
                continue
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if state[ny, nx] == STATE_VISIBLE and 0 < content[ny, nx] < CONTENT_MINE:
                            border[y, x] = True
    return border


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()

    # Imported here so the file can be read without torch present.
    import importlib.util

    source = (Path(__file__).parent / "dataset.py").read_text()
    source = source.replace(
        "import torch\nfrom torch.utils.data import Dataset", "Dataset = object"
    )
    namespace = {}
    exec(compile(source, "dataset.py", "exec"), namespace)
    patch_from_planes = namespace["patch_from_planes"]

    prefix = args.data_dir / args.split
    boards = np.load(f"{prefix}_boards.npy", mmap_mode="r")
    cells = np.load(f"{prefix}_cells.npy")
    labels = np.load(f"{prefix}_labels.npy")
    ratios = np.load(f"{prefix}_ratios.npy")
    records = [json.loads(line) for line in open(args.data_dir / f"{args.split}.jsonl")]

    if len(records) != len(boards):
        raise SystemExit(
            f"{len(records)} records in the JSONL but {len(boards)} prepared boards — "
            "rerun prepare.py"
        )

    rng = np.random.default_rng(0)
    chosen = rng.choice(len(cells), size=min(args.samples, len(cells)), replace=False)
    for idx in chosen:
        record_index, cx, cy = (int(v) for v in cells[idx])
        record = records[record_index]
        height, width = record["height"], record["width"]
        state = np.array(
            [[record["grid"][y][x]["state"] for x in range(width)] for y in range(height)],
            dtype=np.uint8,
        )
        content = np.array(
            [[record["grid"][y][x]["content"] for x in range(width)] for y in range(height)],
            dtype=np.uint8,
        )

        if state[cy, cx] != STATE_HIDDEN:
            raise SystemExit(f"sample {idx} points at a cell that is not hidden")

        hidden = int((state == STATE_HIDDEN).sum())
        mines_left = record["mines_count"] - int((content == CONTENT_MINE).sum())
        ratio = mines_left / max(hidden, 1)
        if abs(ratio - float(ratios[record_index])) > 1e-6:
            raise SystemExit(f"record {record_index}: mines ratio does not match")

        expected = reference_patch(
            state, content, reference_border(state, content), cx, cy, ratio
        )
        board = boards[record_index]
        actual = patch_from_planes(board[0], board[1], board[2], cx, cy, ratio)
        if not np.array_equal(expected, actual):
            differing = sorted({int(c) for c in np.argwhere(expected != actual)[:, 0]})
            raise SystemExit(
                f"record {record_index}, cell ({cx}, {cy}): channels {differing} differ"
            )

        # Labels are stored as float32, which is what the model trains in, so the
        # tolerance has to be float32-sized. A tighter one flags ordinary rounding:
        # 0.2 comes back as 0.20000000298, which is 3e-9 out.
        if abs(float(labels[idx]) - record["probs"][cy][cx]) > 1e-6:
            raise SystemExit(
                f"sample {idx}: label {labels[idx]!r} does not match the record's "
                f"{record['probs'][cy][cx]!r}"
            )

    print(f"OK: {len(chosen)} patches and labels match the JSONL they came from")


if __name__ == "__main__":
    main()
