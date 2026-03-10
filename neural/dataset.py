"""MinesweeperDataset and patch extraction for PatchCNN training.

Each JSONL record contains one game state with ConstraintSearch probability labels.
For each hidden cell we extract a 9×9 patch (8 channels) centred on that cell.

Channel layout (HALF=4, patch size 9×9):
  0: mine_count/8.0 if Visible, else 0
  1: 1 if Visible
  2: 1 if Hidden
  3: 1 if Flagged
  4: 1 if out-of-bounds (padding)
  5: 1 at centre (4,4) else 0  — target cell marker
  6: mines_remaining / total_hidden  — broadcast global ratio
  7: 1 if the patch cell is a "border" hidden cell (adjacent to a visible number)

D4 augmentation is applied randomly per __getitem__ (4 rotations × 2 flips).
Labels are rotation/flip invariant.
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

HALF = 4
PATCH = 2 * HALF + 1  # 9
N_CHANNELS = 8

# State codes from datagen
STATE_HIDDEN = 0
STATE_VISIBLE = 1
STATE_FLAGGED = 2

# Content sentinel for non-visible cells
CONTENT_HIDDEN_SENTINEL = 255
CONTENT_MINE = 9


def _border_mask(grid_state: np.ndarray, grid_content: np.ndarray) -> np.ndarray:
    """Return bool array: True for hidden/flagged cells adjacent to a visible number."""
    h, w = grid_state.shape
    vis_num = (grid_state == STATE_VISIBLE) & (grid_content < CONTENT_MINE) & (grid_content > 0)
    border = np.zeros((h, w), dtype=bool)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(vis_num, dy, axis=0), dx, axis=1)
            # Zero out wrapped edges
            if dy > 0:
                shifted[:dy, :] = False
            elif dy < 0:
                shifted[dy:, :] = False
            if dx > 0:
                shifted[:, :dx] = False
            elif dx < 0:
                shifted[:, dx:] = False
            border |= shifted
    # Only hidden/flagged cells are borders
    border &= (grid_state == STATE_HIDDEN) | (grid_state == STATE_FLAGGED)
    return border


def extract_patch_tensor(
    grid_state: np.ndarray,   # (H, W) uint8 state codes
    grid_content: np.ndarray, # (H, W) uint8 content codes
    cx: int,
    cy: int,
    mines_ratio: float,
    border_mask: np.ndarray,  # (H, W) bool
) -> np.ndarray:
    """Return float32 array of shape (N_CHANNELS, PATCH, PATCH)."""
    h, w = grid_state.shape
    patch = np.zeros((N_CHANNELS, PATCH, PATCH), dtype=np.float32)

    for pi in range(PATCH):
        for pj in range(PATCH):
            gy = cy + pi - HALF
            gx = cx + pj - HALF
            if gy < 0 or gy >= h or gx < 0 or gx >= w:
                patch[4, pi, pj] = 1.0  # out-of-bounds
                continue
            s = grid_state[gy, gx]
            c = grid_content[gy, gx]
            if s == STATE_VISIBLE:
                patch[1, pi, pj] = 1.0
                if c < CONTENT_MINE:
                    patch[0, pi, pj] = c / 8.0
            elif s == STATE_HIDDEN:
                patch[2, pi, pj] = 1.0
            elif s == STATE_FLAGGED:
                patch[3, pi, pj] = 1.0
            if border_mask[gy, gx]:
                patch[7, pi, pj] = 1.0

    # Centre marker
    patch[5, HALF, HALF] = 1.0
    # Global ratio broadcast
    patch[6, :, :] = mines_ratio

    return patch


def _d4_transform(patch: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """Apply D4 symmetry: k rotations of 90°, optional horizontal flip."""
    patch = np.rot90(patch, k, axes=(1, 2))
    if flip:
        patch = np.flip(patch, axis=2)
    return np.ascontiguousarray(patch)


class MinesweeperDataset(Dataset):
    """Dataset that yields (patch_tensor, label) pairs from a JSONL file.

    Each game record is expanded into one entry per hidden cell.
    Indices are built lazily on first access.
    """

    def __init__(self, jsonl_path: str, augment: bool = True):
        self.path = Path(jsonl_path)
        self.augment = augment
        # Each entry: (line_index, cx, cy)
        self._index: Optional[list] = None
        self._lines: Optional[list] = None

    def _build_index(self):
        self._lines = []
        self._index = []
        with open(self.path) as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                h, w = rec["height"], rec["width"]
                for y in range(h):
                    for x in range(w):
                        if rec["grid"][y][x]["state"] == STATE_HIDDEN:
                            self._index.append((len(self._lines), x, y))
                self._lines.append(rec)

    def __len__(self):
        if self._index is None:
            self._build_index()
        return len(self._index)

    def __getitem__(self, idx):
        if self._index is None:
            self._build_index()

        line_idx, cx, cy = self._index[idx]
        rec = self._lines[line_idx]
        h, w = rec["height"], rec["width"]

        grid_state = np.array(
            [[rec["grid"][y][x]["state"] for x in range(w)] for y in range(h)],
            dtype=np.uint8,
        )
        grid_content = np.array(
            [[rec["grid"][y][x]["content"] for x in range(w)] for y in range(h)],
            dtype=np.uint8,
        )

        # mines_remaining / total_hidden
        total_hidden = int((grid_state == STATE_HIDDEN).sum())
        mines_remaining = rec["mines_count"] - int(
            (grid_content == CONTENT_MINE).sum()
        )
        mines_ratio = mines_remaining / max(total_hidden, 1)

        bmask = _border_mask(grid_state, grid_content)
        patch = extract_patch_tensor(grid_state, grid_content, cx, cy, mines_ratio, bmask)

        if self.augment:
            k = random.randint(0, 3)
            flip = random.random() < 0.5
            patch = _d4_transform(patch, k, flip)

        label = float(rec["probs"][cy][cx])
        return torch.from_numpy(patch), torch.tensor(label, dtype=torch.float32)
