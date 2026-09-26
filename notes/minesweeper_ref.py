"""A readable Python twin of the Rust estimator, for the notebooks beside it.

This is not the engine. The engine is `minesweeper_core`, in Rust, and it is the
thing that ships. This file exists so the two notes in this directory can *run*
what they describe — every number in them is computed, not quoted — and so the
algorithm can be read without reading Rust.

It follows `minesweeper_core/src/probability/` step for step and is checked
against a brute-force oracle at the bottom of `constraint-estimator.ipynb`.
Where the Rust is careful for reasons of scale (rescaled binomials, a node
budget, a cross-move cache) this keeps the same structure but not the same
tuning: it is meant to be read.

numpy is the only dependency.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

HIDDEN, VISIBLE, FLAGGED = 0, 1, 2

# --------------------------------------------------------------------- board


@dataclass
class Board:
    """A position: where the mines are, and what the player has been shown."""

    width: int
    height: int
    mines: set[tuple[int, int]]
    state: dict[tuple[int, int], int] = field(default_factory=dict)

    def __post_init__(self):
        for y in range(self.height):
            for x in range(self.width):
                self.state.setdefault((x, y), HIDDEN)

    @property
    def mines_count(self) -> int:
        return len(self.mines)

    def neighbours(self, x: int, y: int):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    yield nx, ny

    def number(self, x: int, y: int) -> int:
        """How many mines touch this square."""
        return sum((nx, ny) in self.mines for nx, ny in self.neighbours(x, y))

    def reveal(self, x: int, y: int):
        """Open a square, cascading through zeros the way the game does."""
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if self.state[(cx, cy)] == VISIBLE:
                continue
            self.state[(cx, cy)] = VISIBLE
            if (cx, cy) not in self.mines and self.number(cx, cy) == 0:
                stack.extend(self.neighbours(cx, cy))
        return self

    def flag(self, x: int, y: int):
        self.state[(x, y)] = FLAGGED
        return self

    def hidden_cells(self) -> list[tuple[int, int]]:
        """Unopened squares, row-major — the order every index below refers to."""
        return [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.state[(x, y)] != VISIBLE
        ]

    def render(self, values: dict[tuple[int, int], float] | None = None, width: int = 5) -> str:
        """The board as text, optionally with a number written in each unopened cell."""
        rows = []
        for y in range(self.height):
            cells = []
            for x in range(self.width):
                if self.state[(x, y)] == VISIBLE:
                    n = self.number(x, y)
                    cells.append(("*" if (x, y) in self.mines else (str(n) if n else "·")).rjust(width))
                elif values is not None and (x, y) in values:
                    cells.append(f"{values[(x, y)] * 100:.0f}%".rjust(width))
                else:
                    cells.append(("F" if self.state[(x, y)] == FLAGGED else "▒").rjust(width))
            rows.append("".join(cells))
        return "\n".join(rows)


def from_text(text: str) -> Board:
    """Build a position from a picture of it.

    `*` is a mine, `#` a mine the player has flagged, `.` an unopened empty
    square, and anything else — `o` by convention — an opened one. The numbers
    are computed from the mine positions, not read from the picture, so a picture
    cannot disagree with itself.
    """
    rows = [row for row in text.strip("\n").split("\n") if row.strip()]
    height, width = len(rows), max(len(row) for row in rows)
    mines, opened, flagged = set(), set(), set()
    for y, row in enumerate(rows):
        for x, char in enumerate(row.ljust(width)):
            if char == "*":
                mines.add((x, y))
            elif char == "#":
                mines.add((x, y))
                flagged.add((x, y))
            elif char not in ".":
                opened.add((x, y))
    board = Board(width, height, mines)
    for cell in opened:
        board.state[cell] = VISIBLE
    for cell in flagged:
        board.state[cell] = FLAGGED
    return board


# ---------------------------------------------------------------- constraints


def build_constraints(board: Board):
    """Turn every visible number into "exactly k of these unopened cells".

    Mirrors `SimSetup::build`. Returns the hidden cells (the index space for
    everything downstream), the constraints over them, and how many mines are
    still out there.
    """
    hidden = board.hidden_cells()
    index = {cell: i for i, cell in enumerate(hidden)}

    # A mine is only ever visible on a board that has been lost, but the arithmetic
    # has to hold there too: such a mine is accounted for, so it comes off both
    # the numbers that see it and the total still to place.
    visible_mines = sum(1 for c in board.mines if board.state[c] == VISIBLE)

    constraints = []
    for y in range(board.height):
        for x in range(board.width):
            if board.state[(x, y)] != VISIBLE or (x, y) in board.mines:
                continue
            cells = [index[n] for n in board.neighbours(x, y) if n in index]
            seen = sum(
                1 for n in board.neighbours(x, y)
                if n in board.mines and board.state[n] == VISIBLE
            )
            if cells:
                # A flagged cell is still unknown to the solver: the player's
                # flag is an opinion, not evidence, so it stays in the constraint.
                constraints.append((sorted(cells), board.number(x, y) - seen))

    return hidden, constraints, board.mines_count - visible_mines


# --------------------------------------------------------------- propagation


def propagate(n: int, constraints, mines_to_place: int):
    """What local rules alone prove, to a fixpoint. Mirrors `propagate`.

    Two rules, applied until nothing changes: a constraint whose mines are all
    accounted for makes its other cells safe, and one with exactly as many
    undecided cells as mines left makes them all mines. The board's total mine
    count is the same rule applied globally.

    Sound but incomplete — "not proven" never means "not certain".
    """
    is_mine = [False] * n
    is_safe = [False] * n

    changed = True
    while changed:
        changed = False
        for cells, required in constraints:
            known = sum(is_mine[i] for i in cells)
            undecided = [i for i in cells if not is_mine[i] and not is_safe[i]]
            remaining = max(required - known, 0)
            if remaining == 0:
                for i in undecided:
                    is_safe[i], changed = True, True
            elif remaining == len(undecided):
                for i in undecided:
                    is_mine[i], changed = True, True

        undecided = [i for i in range(n) if not is_mine[i] and not is_safe[i]]
        left = max(mines_to_place - sum(is_mine), 0)
        if left == 0 and undecided:
            for i in undecided:
                is_safe[i], changed = True, True
        elif left == len(undecided) and undecided:
            for i in undecided:
                is_mine[i], changed = True, True

    mines = [i for i in range(n) if is_mine[i]]
    safe = [i for i in range(n) if is_safe[i]]

    reduced = []
    for cells, required in constraints:
        rest = [i for i in cells if not is_mine[i] and not is_safe[i]]
        if rest:
            reduced.append((rest, required - sum(is_mine[i] for i in cells)))

    return mines, safe, reduced, mines_to_place - len(mines)


# ------------------------------------------------------------- decomposition


def decompose(constraints, n: int):
    """Split the constraint graph into parts that share no cell.

    Union-find over "appears in the same constraint", exactly as `decompose`
    does. Two parts are independent apart from the board's total mine count,
    which is what makes solving them separately valid.
    """
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for cells, _ in constraints:
        for cell in cells[1:]:
            union(cells[0], cell)

    groups: dict[int, list] = {}
    for cells, required in constraints:
        groups.setdefault(find(cells[0]), []).append((cells, required))

    components = []
    for group in groups.values():
        cells = sorted({c for cells, _ in group for c in cells})
        local = {c: i for i, c in enumerate(cells)}
        components.append(
            (cells, [([local[c] for c in cs], required) for cs, required in group])
        )
    components.sort(key=lambda comp: comp[0][0])
    return components


# ------------------------------------------------------------ solving a part


def solve_component(cells, constraints):
    """Count this part's layouts, indexed by how many mines they use.

    Returns `ways[k]` and `cell_ways[c][k]`. Deliberately *not* probabilities:
    how likely a layout is depends on how many mines the rest of the board
    takes, but these counts do not — which is what lets parts be combined, and
    cached across moves.

    The Rust does a pruned depth-first walk with a node budget. This enumerates
    the subsets outright, which is clearer and fine at notebook scale.
    """
    size = len(cells)
    ways = [0] * (size + 1)
    cell_ways = [[0] * (size + 1) for _ in range(size)]

    for bits in itertools.product((0, 1), repeat=size):
        if any(sum(bits[i] for i in cs) != required for cs, required in constraints):
            continue
        k = sum(bits)
        ways[k] += 1
        for c, bit in enumerate(bits):
            if bit:
                cell_ways[c][k] += 1

    return ways, cell_ways


def convolve(a, b):
    """Layout counts of two independent parts, combined by mine count."""
    out = [0.0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def binomials(n: int):
    """C(n, k) for every k. The Rust rescales these in log space because on a
    real board they overflow f64; at notebook scale exact integers are clearer."""
    return [math.comb(n, k) for k in range(n + 1)]


# ------------------------------------------------------------------ combining


def combine(hidden, components, solutions, interior, mines_to_place):
    """Put the parts back together and read off per-cell probabilities.

    The number of whole-board layouts using `b` border mines is the convolution
    of the parts; the interior — cells no number speaks about — takes whatever
    is left, in C(interior, left) ways. A cell's probability is the share of
    that total in which it is a mine.
    """
    n = len(hidden)
    interior_ways = binomials(len(interior))

    def weight(border_mines: int) -> float:
        left = mines_to_place - border_mines
        return interior_ways[left] if 0 <= left <= len(interior) else 0.0

    # prefix[j] convolves parts 0..j, suffix[j] convolves j.. — so everything
    # except part j is prefix[j] * suffix[j+1]. Built this way rather than by
    # dividing the total out, which breaks wherever a part contributes a zero.
    prefix = [[1.0]]
    for ways, _ in solutions:
        prefix.append(convolve(prefix[-1], ways))
    suffix = [[1.0]]
    for ways, _ in reversed(solutions):
        suffix.append(convolve(suffix[-1], ways))
    suffix.reverse()

    total_ways = prefix[-1]
    total = sum(w * weight(b) for b, w in enumerate(total_ways))
    if total <= 0:
        return None

    probs = [0.0] * n
    for j, ((cells, _), (ways, cell_ways)) in enumerate(zip(components, solutions)):
        rest = convolve(prefix[j], suffix[j + 1])
        share = [
            sum(w * weight(b + k) for b, w in enumerate(rest)) for k in range(len(ways))
        ]
        for local, cell in enumerate(cells):
            numerator = sum(w * s for w, s in zip(cell_ways[local], share))
            # Certainty is decided from the integer counts, never from the ratio.
            # A rescaled weight can underflow to zero in the tails, and a zero
            # read off the ratio would claim the cell is *provably* safe — which
            # every caller acts on by opening it.
            if all(w == 0 for w in cell_ways[local]):
                probs[cell] = 0.0
            elif all(m == t for m, t in zip(cell_ways[local], ways)):
                probs[cell] = 1.0
            else:
                probs[cell] = numerator / total

    if interior:
        numerator = 0.0
        for b, w in enumerate(total_ways):
            left = mines_to_place - b
            if 0 <= left <= len(interior):
                numerator += w * weight(b) * left / len(interior)
        value = numerator / total
        for cell in interior:
            probs[cell] = value

    return probs


def exact_probabilities(board: Board) -> dict[tuple[int, int], float]:
    """The whole pipeline: constraints, propagation, decomposition, combination."""
    hidden, constraints, mines_to_place = build_constraints(board)
    n = len(hidden)
    if n == 0:
        return {}

    mines, safe, reduced, left = propagate(n, constraints, mines_to_place)

    decided = set(mines) | set(safe)
    free = [i for i in range(n) if i not in decided]
    remap = {c: i for i, c in enumerate(free)}
    reduced = [([remap[c] for c in cs], r) for cs, r in reduced]

    components = decompose(reduced, len(free))
    solutions = [solve_component(cells, cons) for cells, cons in components]
    in_a_component = {c for cells, _ in components for c in cells}
    interior = [i for i in range(len(free)) if i not in in_a_component]

    local_probs = combine(free, components, solutions, interior, left)

    out = {hidden[i]: 1.0 for i in mines}
    out.update({hidden[i]: 0.0 for i in safe})
    if local_probs is not None:
        out.update({hidden[free[i]]: p for i, p in enumerate(local_probs)})
    return out


# -------------------------------------------------------------------- oracle


def brute_force(board: Board) -> dict[tuple[int, int], float]:
    """Every consistent layout, counted directly. Correct, and hopeless past ~20
    unopened cells — which is exactly why the rest of this file exists."""
    hidden, constraints, mines_to_place = build_constraints(board)
    n = len(hidden)
    counts = [0] * n
    total = 0

    for positions in itertools.combinations(range(n), mines_to_place):
        layout = [False] * n
        for i in positions:
            layout[i] = True
        if any(sum(layout[i] for i in cs) != required for cs, required in constraints):
            continue
        total += 1
        for i in positions:
            counts[i] += 1

    if total == 0:
        return {}
    return {hidden[i]: counts[i] / total for i in range(n)}


# ---------------------------------------------------- the network's input

HALF, PATCH, N_CHANNELS = 4, 9, 8


def patch(board: Board, cx: int, cy: int) -> np.ndarray:
    """The 9x9x8 window the network is asked about.

    Mirrors `probability/patch.rs` and `neural/dataset.py`. The layout is written
    down in three places and nothing ties them together, so a disagreement here
    does not raise — it just feeds the model inputs it never saw.
    """
    out = np.zeros((N_CHANNELS, PATCH, PATCH), dtype=np.float32)

    # Both halves of this are easy to get subtly wrong, and getting them wrong
    # does not raise — it just shifts one input channel. The denominator counts
    # only *Hidden* cells, not flagged ones, and the numerator subtracts mines
    # the player has actually uncovered, not flags they have guessed at.
    hidden_only = sum(1 for c in board.state if board.state[c] == HIDDEN)
    visible_mines = sum(
        1 for c in board.mines if board.state[c] == VISIBLE
    )
    ratio = (board.mines_count - visible_mines) / max(hidden_only, 1)

    def touches_a_number(x, y):
        # A visible *zero* does not make a neighbour a border cell: it says
        # nothing that constrains anything.
        return any(
            board.state[n] == VISIBLE and n not in board.mines and board.number(*n) > 0
            for n in board.neighbours(x, y)
        )

    for row in range(PATCH):
        for col in range(PATCH):
            x, y = cx - HALF + col, cy - HALF + row
            if not (0 <= x < board.width and 0 <= y < board.height):
                out[4, row, col] = 1.0          # off the edge
                continue
            state = board.state[(x, y)]
            if state == VISIBLE:
                out[1, row, col] = 1.0
                if (x, y) not in board.mines:
                    out[0, row, col] = board.number(x, y) / 8.0
            elif state == FLAGGED:
                out[3, row, col] = 1.0
            else:
                out[2, row, col] = 1.0
            if state != VISIBLE and touches_a_number(x, y):
                out[7, row, col] = 1.0

    out[5, HALF, HALF] = 1.0                    # this is the cell being asked about
    out[6, :, :] = ratio
    return out


# -------------------------------------------------------------- the network


class Network:
    """The trained PatchCNN, loaded once and run in batch.

    `neural/patchcnn_reference.py` is the specification — loops where a library
    call would do, so that it is transparently correct. It also re-reads the
    weights file on every call, which is 150 ms a cell. This is the same
    arithmetic vectorised, for the measurements in the notebook; it is checked
    against the reference there, cell by cell.

    Four 3x3 convolutions with ReLU, a 3x3 average pool, the global mines ratio
    appended back on (it would otherwise be blurred away by the pooling), then
    128 -> 64 -> 1 and a sigmoid. BatchNorm has been folded into the preceding
    convolution by the exporter, so there is none here to apply.
    """

    def __init__(self, path: str = "../neural/onnx/model.bin"):
        from struct import unpack  # noqa: F401  (documented layout, see below)

        MAGIC = b"MSPCNN1\0"
        raw = open(path, "rb").read()
        if raw[: len(MAGIC)] != MAGIC:
            raise ValueError(f"{path} does not start with {MAGIC!r}")
        values = np.frombuffer(raw, dtype="<f4", offset=len(MAGIC))

        at = 0

        def take(*shape):
            nonlocal at
            count = int(np.prod(shape))
            out = values[at : at + count].reshape(shape)
            at += count
            return out

        self.convs = [(take(o, i, 3, 3), take(o)) for o, i in
                      [(32, 8), (64, 32), (64, 64), (32, 64)]]
        self.linears = [(take(o, i), take(o)) for o, i in
                        [(128, 289), (64, 128), (1, 64)]]
        if at != len(values):
            raise ValueError(f"{path} holds {len(values)} values, the layout wants {at}")

    @property
    def parameters(self) -> int:
        return sum(w.size + b.size for w, b in self.convs + self.linears)

    @staticmethod
    def _convolve(x, weight, bias):
        """3x3, stride 1, one cell of zero padding — as nine shifted views."""
        channels, height, width = x.shape
        padded = np.zeros((channels, height + 2, width + 2), dtype=np.float32)
        padded[:, 1:-1, 1:-1] = x
        windows = np.stack(
            [padded[:, ky : ky + height, kx : kx + width] for ky in range(3) for kx in range(3)]
        ).reshape(3, 3, channels, height, width)
        return np.einsum("ocyx,yxchw->ohw", weight, windows) + bias[:, None, None]

    def features(self, patch: np.ndarray) -> np.ndarray:
        """Everything up to the head — what the last layer sees."""
        x = np.asarray(patch, dtype=np.float32)
        for weight, bias in self.convs:
            x = np.maximum(self._convolve(x, weight, bias), 0.0)
        pooled = x.reshape(x.shape[0], 3, 3, 3, 3).mean(axis=(2, 4))
        head = np.concatenate([pooled.reshape(-1), [patch[6][4][4]]]).astype(np.float32)
        for weight, bias in self.linears[:-1]:
            head = np.maximum(weight @ head + bias, 0.0)   # dropout is identity at eval
        return head

    def predict(self, patch: np.ndarray) -> float:
        weight, bias = self.linears[-1]
        return float(1.0 / (1.0 + np.exp(-(weight @ self.features(patch) + bias)[0])))

    def learn(self, patch: np.ndarray, target: float, rate: float) -> float:
        """One gradient step on the output layer only, as the page does in play.

        With a sigmoid output and cross-entropy loss the gradient into the head's
        weights is just `(prediction - target) * feature` — no chain rule left to
        apply, which is why this is a dozen lines of Rust in the browser. The
        forward pass it needs is the same one `predict` does, which is why the
        page folds correction into the scoring pass instead of running a second.

        Returns the prediction *before* the step.
        """
        features = self.features(patch)
        weight, bias = self.linears[-1]
        before = float(1.0 / (1.0 + np.exp(-(weight @ features + bias)[0])))
        error = before - target
        self.linears[-1] = (weight - rate * error * features, bias - rate * error)
        return before

    def score_board(self, board: Board) -> dict[tuple[int, int], float]:
        """P(mine) for every unopened cell."""
        return {cell: self.predict(patch(board, *cell)) for cell in board.hidden_cells()}
