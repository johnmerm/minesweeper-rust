"""MinesweeperDataset and patch extraction for PatchCNN training.

Each training sample is one hidden cell, represented by a 9x9 patch centred on it.

Channel layout (HALF=4, patch size 9x9):
  0: mine_count/8.0 if Visible, else 0
  1: 1 if Visible
  2: 1 if Hidden
  3: 1 if Flagged
  4: 1 if out-of-bounds (padding)
  5: 1 at centre (4,4) else 0  — target cell marker
  6: mines_remaining / total_hidden  — broadcast global ratio
  7: 1 if the patch cell is a "border" hidden cell (adjacent to a visible number)

D4 augmentation is applied randomly per __getitem__ (4 rotations x 2 flips).
Labels are rotation/flip invariant.

# Reading from prepared arrays

The dataset reads the `.npy` files written by `prepare.py`, memory-mapped, rather
than JSONL. This is not a micro-optimisation: parsing JSONL per epoch meant
holding every record in memory as Python dicts, which for the 500 000 samples the
workflow suggests is about 11.5 GB before `DataLoader` forks it once per worker.
That is why training never got off the ground. Memory-mapped uint8 planes cost
about 380 MB of disk for the same data and near-zero resident memory, and the
operating system pages in only what is being read.

Patch extraction is a slice of three preprocessed planes instead of a Python loop
over 81 positions with the border mask recomputed per cell.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HALF = 4
PATCH = 2 * HALF + 1  # 9
N_CHANNELS = 8

# State codes from datagen, plus the one prepare.py uses for padding.
STATE_HIDDEN = 0
STATE_VISIBLE = 1
STATE_FLAGGED = 2
STATE_OUT_OF_BOUNDS = 3

CONTENT_MINE = 9

# Plane indices within a prepared board.
PLANE_STATE = 0
PLANE_CONTENT = 1
PLANE_BORDER = 2


def patch_from_planes(
    state: np.ndarray,
    content: np.ndarray,
    border: np.ndarray,
    cx: int,
    cy: int,
    mines_ratio: float,
) -> np.ndarray:
    """Build the 8-channel patch centred on `(cx, cy)`.

    `state` outside the board must already read `STATE_OUT_OF_BOUNDS`, which
    `prepare.py` arranges, so the window can be taken by slicing and only the part
    that falls off the array edge needs filling in.
    """
    height, width = state.shape
    window_state = np.full((PATCH, PATCH), STATE_OUT_OF_BOUNDS, dtype=np.uint8)
    window_content = np.zeros((PATCH, PATCH), dtype=np.uint8)
    window_border = np.zeros((PATCH, PATCH), dtype=np.uint8)

    top, left = cy - HALF, cx - HALF
    y0, y1 = max(top, 0), min(top + PATCH, height)
    x0, x1 = max(left, 0), min(left + PATCH, width)
    if y0 < y1 and x0 < x1:
        to = (slice(y0 - top, y1 - top), slice(x0 - left, x1 - left))
        window_state[to] = state[y0:y1, x0:x1]
        window_content[to] = content[y0:y1, x0:x1]
        window_border[to] = border[y0:y1, x0:x1]

    patch = np.zeros((N_CHANNELS, PATCH, PATCH), dtype=np.float32)
    visible = window_state == STATE_VISIBLE
    numbered = visible & (window_content < CONTENT_MINE)
    patch[0][numbered] = window_content[numbered] / 8.0
    patch[1][visible] = 1.0
    patch[2][window_state == STATE_HIDDEN] = 1.0
    patch[3][window_state == STATE_FLAGGED] = 1.0
    patch[4][window_state == STATE_OUT_OF_BOUNDS] = 1.0
    patch[5, HALF, HALF] = 1.0
    patch[6, :, :] = mines_ratio
    patch[7][window_border.astype(bool)] = 1.0
    return patch


def _d4_transform(patch: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """Apply D4 symmetry: k rotations of 90 degrees, optional horizontal flip."""
    patch = np.rot90(patch, k, axes=(1, 2))
    if flip:
        patch = np.flip(patch, axis=2)
    return np.ascontiguousarray(patch)


class MinesweeperDataset(Dataset):
    """Yields (patch_tensor, label) pairs from the arrays `prepare.py` writes.

    `prefix` is the split's path without the suffix, e.g. `data/train`, so that
    `data/train_boards.npy` and friends are what get opened.
    """

    def __init__(self, prefix, augment: bool = True):
        prefix = Path(prefix)
        missing = [
            name
            for name in ("boards", "cells", "labels", "ratios")
            if not prefix.with_name(f"{prefix.name}_{name}.npy").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{prefix}_{{{','.join(missing)}}}.npy not found — run "
                f"`python neural/prepare.py` to convert the JSONL from datagen first"
            )

        self.augment = augment
        # Memory-mapped: the boards are far larger than the index and are read in
        # a shuffled order, so let the operating system decide what to keep.
        self.boards = np.load(f"{prefix}_boards.npy", mmap_mode="r")
        self.cells = np.load(f"{prefix}_cells.npy")
        self.labels = np.load(f"{prefix}_labels.npy")
        self.ratios = np.load(f"{prefix}_ratios.npy")

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        record, cx, cy = self.cells[idx]
        board = self.boards[record]
        patch = patch_from_planes(
            board[PLANE_STATE],
            board[PLANE_CONTENT],
            board[PLANE_BORDER],
            int(cx),
            int(cy),
            float(self.ratios[record]),
        )

        if self.augment:
            patch = _d4_transform(patch, random.randint(0, 3), random.random() < 0.5)

        return torch.from_numpy(patch), torch.tensor(self.labels[idx], dtype=torch.float32)
