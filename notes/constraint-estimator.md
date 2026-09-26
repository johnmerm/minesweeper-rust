# How the exact probabilities are computed

*Companion notebook: [`constraint-estimator.ipynb`](constraint-estimator.ipynb) — runs
every step below on real positions and checks the answers against brute force.*

Every unopened cell on the board carries a number: the fraction of consistent mine
layouts in which that cell holds a mine. It is a definition, not an estimate —
count the layouts, count the ones with a mine there, divide. The whole of
`minesweeper_core::probability` exists because that count is astronomically large
and the division has to happen anyway.

This note is about how it is made tractable, and about the two places where being
clever nearly broke it.

---

## The problem

On a 30×16 Expert board after a few moves there might be 300 unopened cells and
99 mines. The number of ways to place them is C(300, 99) ≈ 10⁸². The consistent
ones — those where every visible number sees exactly as many mines as it says —
are far fewer, but still far too many to enumerate.

What saves it is that the constraints are **local and sparse**. A visible `3`
says something about at most eight cells. Most pairs of cells are related by no
constraint at all.

---

## The pipeline

Four stages, in `probability/`:

```
board
  │
  ├─ SimSetup::build ──────► hidden cells, one constraint per visible number
  │
  ├─ propagate ───────────► the cells local rules alone settle  (cheap, incomplete)
  │
  ├─ components::decompose ► independent groups  (this is the important one)
  │
  ├─ ConstraintSearch ─────► per group: ways[k], cell_ways[c][k]
  │
  └─ components::combine ──► convolve the groups, fold in the interior, divide
```

### 1. Constraints

Each visible number becomes *"exactly k of these unopened neighbours are mines"*.
Plus one global constraint: the whole board holds a fixed number of mines.

A flagged cell stays in the constraints as an unknown. A flag is the player's
opinion; the solver has no reason to believe it. (It is excluded from the
*neural* estimator's global ratio for a different reason — see the other note.)

A visible `0` is a constraint too. A cascade normally leaves zeros with no hidden
neighbours, but a board restored from a snapshot carries no such guarantee, so
the code does not rely on it.

### 2. Propagation

Two rules, applied until nothing changes:

- a constraint whose mines are all accounted for makes its remaining cells **safe**;
- a constraint with exactly as many undecided cells as mines left makes them all **mines**.

The global mine count is the same pair of rules applied to the whole board.

This costs no search and settles most of a typical position. It is **sound but
incomplete**: it never marks a cell wrongly, but it misses cells only a full
enumeration can prove. A caller must never read *"not proven"* as *"not certain"*.
`probability::certain_cells` exposes exactly this, and auto-play iterates on it,
paying for a full solve only once it runs dry — re-solving after every pass is
what once made a single click take 31 seconds.

### 3. Decomposition — the one that matters

Two cells are related only if some number counts them both, directly or through
a chain. So the border falls into groups that share no cell. Union-find over
*"appears in the same constraint"* finds them.

The groups are independent **apart from the board's total mine count**. That is
the whole trick, and it turns a product into a sum:

```
one problem:     solutions(A) × solutions(B) × solutions(C)
three problems:  solutions(A) + solutions(B) + solutions(C)
```

A 50×50 board with 150 mines splits into roughly 27 groups per move. This one
change took the worst measured click from **137 seconds to about 12 milliseconds**.

### 4. What a group returns

Not a probability. How likely a group's layouts are depends on how many mines the
*rest* of the board takes, so a probability is not a property of the group.

What is a property of the group — independent of the total mine count and of
every other group — is the count of layouts per mine count:

- `ways[k]` — layouts of this group using exactly `k` mines
- `cell_ways[c][k]` — how many of those have cell `c` as a mine

This form is why groups can be combined at all. It is also why they can be
**cached across moves**: a group's solution is a function of exactly its squares
and the numbers around them, so the cache key is those, and an entry can never go
stale — a board that changes a group changes its key too. Measured reuse after a
single reveal: 97% of groups on 50×50/150, 86% on 50×50/300.

The key is built from **board coordinates**, deliberately. Hidden-cell indices
are renumbered on every move as cells open, so a key built from them would miss
every time — or worse, collide between different squares.

### 5. Combining

The number of whole-board layouts using `b` border mines is the convolution of
the groups' `ways`. The **interior** — cells no number speaks about — takes
whatever mines are left, in `C(interior, left)` ways.

For each group the code needs *"everything except this group"*. It gets it from a
prefix and a suffix convolution, `prefix[j] × suffix[j+1]`, rather than by
dividing the total. Division breaks wherever a group contributes a zero.

Every interior cell has the same probability as every other, since nothing
distinguishes them.

---

## The two traps

Both of these were live bugs. Both produced *wrong answers that looked fine*.

### Scaled binomials

`C(n, k)` overflows `f64` on any real board, so the weights are computed in log
space and rescaled by subtracting a peak. The scale factor cancels because the
same table divides numerator and denominator — **never compare weights across two
tables.**

The first version normalised by the table's *global* peak. On a sparse board the
reachable range of `k` sits far out in the tail, so `exp(566 - 1663)` underflowed
to zero, every weight became zero, and the whole grid read **0%**. Every
front-end reads 0% as proof a cell is safe and opens it. A 30-game sweep caught
it as two detonations on 50×50/150.

The fix: take the peak over the *reachable* `k` range only, plus two guards — a
zero total weight reports no result at all, and auto-play acts only on results
from the exact search.

### A partial search is not a small answer

If a group exhausts its node budget, **the entire answer is discarded** — not
just that group's.

A depth-first walk that stops early has covered a lexicographic prefix of the
search space. A cell whose subtree was never visited reads zero mines out of the
layouts seen, which is 0%, which every caller treats as proof. A partial walk
does not give you a rough answer; it gives you a confident wrong one.

### The invariant that follows

> **Never treat 0.0 or 1.0 as merely a small or large number.**

The solver decides those two values from **integer layout counts**, never from
the computed ratio:

```rust
let never_a_mine = cell_ways.iter().all(|&ways| ways == 0.0);
let always_a_mine = cell_ways.iter().zip(&solution.ways)
    .all(|(&mine, &total)| mine == total);
```

A rescaled weight can underflow; a count cannot lie that way. Everything else is
clamped strictly inside `(0, 1)` so no arithmetic accident can manufacture a
proof.

A **sampled** 0% means only that no draw happened to put a mine there. The Monte
Carlo estimator is a fallback for when the exact search gives up, and nothing
ever auto-opens on its numbers.

---

## Bounds

| | |
|---|---|
| Node budget | `ConstraintSearch::max_nodes`, default 1 000 000, checked at every node |
| Past it | reports no result; caller falls back to Monte Carlo |
| Monte Carlo | bounded too, and bails after `max_attempts / 10` with zero valid samples |
| The fallback | **the open problem** — see *What is left* below |
| Offline | `ConstraintSearch::exhaustive()` removes the bound |
| Worst measured solve | ~12 ms, boards up to 200 a side |

The budget is checked at every node rather than every *n*th valid leaf, because a
deep subtree that never reaches a leaf produces no leaves to count — which is
exactly the subtree that runs away.

---

## What is left

### The requirement

**Every probability the estimator reports must be the true one.** Not only the
0.0 and the 1.0 — all of them. A number a player reads off a cell is a claim
about how the board actually is, and an approximation that looks the same as an
exact answer is a wrong claim dressed as a right one.

Monte Carlo does not meet that bar and is not the goal. It is a stopgap for
positions the exact search refuses, and while it is there the honest position is
that a caller must be able to tell which kind of answer it received. Today it
cannot: `calculate_mine_probabilities` returns a bare `Vec<Vec<f64>>` and the
sampled numbers are indistinguishable from the exact ones. That is the gap to
close, in the API and in every front-end that displays the result.

Exact counting of consistent layouts is #P-hard, so *exact always* and *fast
always* cannot both be promised in the worst case. The way to hold the
requirement anyway is:

1. make the exact method cover so much that the refusal is vanishingly rare, and
2. when it does refuse, **say so** — never substitute a sample and present it as
   the answer.

### What that costs today

Measured on 30x30/250, playing with `certain_cells` and opening the lowest cell
when propagation runs dry: 663 solves, **3 refused by the node budget (0.5%)**.
Each of those is solvable exactly — they need 1.2M to 3.8M nodes against the 1M
budget, and cost 86 ms, 86 ms and 251 ms with the budget removed. One position
found in an earlier sweep needed 10M nodes and 540 ms.

So the refusals are not intractable positions. They are positions a few times
past an arbitrary line, and the exact answer is a few hundred milliseconds away.

The shipped page refuses more often than that, because it reuses a warm cache
and plays differently. Driving `docs/minesweeper.wasm` through 25 games per
configuration, counting every `ms_compute`:

| board | solves | sampled | worst solve |
|---|---|---|---|
| 30x16/99 | 190 | 1 (0.5%) | 148 ms |
| 30x30/250 | 148 | 8 (5.4%) | 224 ms |
| 40x40/400 | 274 | 13 (4.7%) | 245 ms |

So roughly one move in twenty on a dense board currently shows sampled numbers
that look exactly like exact ones. Six games gave 27.6% on the middle row, which
is noise — measure enough games before tuning to a figure.

### The next step

A dynamic program over the search frontier: merge partial assignments that agree
on the cells still in play and on how many mines they have used, instead of
walking each to a leaf. It is exact — it counts the same layouts, it just stops
re-deriving shared suffixes — and polynomial where the border is thin, which it
usually is. It would subsume decomposition rather than replace it, since
disconnected groups are just frontiers that never meet.

Raising `max_nodes` is the cruder version of the same goal and buys most of the
measured gap; it needs the work moved off the click first, the way the neural
scoring pass already is, or a few hundred milliseconds lands on the main thread.


---

## Source

| What | Where |
|---|---|
| Constraints, propagation, Monte Carlo | `minesweeper_core/src/probability/monte_carlo.rs` |
| Decomposition, combination, caching | `minesweeper_core/src/probability/components.rs` |
| The depth-first search and its budget | `minesweeper_core/src/probability/constraint_search.rs` |
| `certain_cells` | `minesweeper_core/src/probability/mod.rs` |
| Tests, including a brute-force oracle | `minesweeper_core/tests/probability.rs` |
| The Python twin used by the notebook | `notes/minesweeper_ref.py` |
