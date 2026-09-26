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

There is no sampled estimator to confuse this with any more. Monte Carlo was
deleted: over 71 mid-game positions the exact search answered all 71 in 20 ms in
total while sampling answered 20 and took 4.4 seconds, and among its answers was
a cell it called 0% that was not safe. When the exact search cannot finish, the
API returns `None` and the front-ends show `?`.

---

## Bounds

| | |
|---|---|
| Node budget | `ConstraintSearch::max_nodes`, default 4 000 000, checked at every node |
| Past it | returns `None`; the caller shows `?` and no colour |
| What is left | **the open problem** — see *What is left* below |
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

Monte Carlo used to fill the gap and has been **deleted**. It could not meet
that bar, and measured against the decomposing exact search it could not even
beat it on speed. Over 71 mid-game positions:

| | exact | Monte Carlo |
|---|---|---|
| answered | 71 of 71 | 20 of 71 |
| total time | 20 ms | 4 449 ms |
| worst error where both answered | — | 0.322 |
| cells called 0% that were not safe | — | 1 |

That last row is the one that settles it. A sampled 0% is read by every
front-end as proof, and on 30x16/99 it was wrong.

So the API says so now: `ProbabilityStrategy::calculate` and
`Minesweeper::calculate_mine_probabilities` return `Option<Vec<Vec<f64>>>`, and
`None` means *no answer*. Nothing substitutes for it. In the wasm ABI the same
fact is `STAT_SOLVED`, and the page draws a distinct colour and a `?` rather
than percentages — `probColor(0)` is the grey that means "certainly safe", so an
unsolved board left untinted would read as a board with no mines on it.

Exact counting of consistent layouts is #P-hard, so *exact always* and *fast
always* cannot both be promised. What is promised is that a number on screen is
correct, and that a position which cannot be solved says so.

### What is still refused, and why the budget cannot fix it

Driving `docs/minesweeper.wasm` through 25 games per configuration, counting
every `ms_compute`:

| board | solves | unsolved | worst solve |
|---|---|---|---|
| 30x16/99 | 190 | 1 (0.5%) | 78 ms |
| 30x30/250 | 148 | 8 (5.4%) | 127 ms |
| 40x40/400 | 274 | 13 (4.7%) | 245 ms |

Raising `DEFAULT_MAX_NODES` is the obvious idea and it does not work. On
30x30/250 the *same 8* positions are refused at 1M, 4M, 8M and 32M nodes — the
budget buys nothing there and the worst solve grows from 127 ms to 2 935 ms,
because the search spends all of it before giving up. Those positions are not a
few times past an arbitrary line; they are genuinely large. Measure before
touching this constant: an earlier run on an easier trajectory suggested every
refusal was within reach, and it was not.

### The next step

A dynamic program over the search frontier: merge partial assignments that agree
on the cells still in play and on how many mines they have used, instead of
walking each to a leaf. It is exact — it counts the same layouts, it just stops
re-deriving shared suffixes — and polynomial where the border is thin, which it
usually is. It would subsume decomposition rather than replace it, since
disconnected groups are just frontiers that never meet.

That, not a bigger budget, is what closes the remaining few percent.
---

## Source

| What | Where |
|---|---|
| Constraints and propagation | `minesweeper_core/src/probability/setup.rs` |
| Decomposition, combination, caching | `minesweeper_core/src/probability/components.rs` |
| The depth-first search and its budget | `minesweeper_core/src/probability/constraint_search.rs` |
| `certain_cells` | `minesweeper_core/src/probability/mod.rs` |
| Tests, including a brute-force oracle | `minesweeper_core/tests/probability.rs` |
| The Python twin used by the notebook | `notes/minesweeper_ref.py` |
