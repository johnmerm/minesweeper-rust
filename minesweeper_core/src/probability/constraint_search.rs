use std::collections::HashSet;
use std::sync::mpsc::Sender;

use crate::Minesweeper;

use super::{ProbabilityStrategy, SimUpdate, Strategy};
use super::monte_carlo::{SimSetup, build_probs, mc_memory_estimate};

/// Exact mine probability estimation using depth-first constraint enumeration.
///
/// # High-level idea
///
/// Every visible numbered cell gives us a *constraint*: "exactly N of my hidden
/// neighbours are mines."  Instead of sampling random boards, we walk every
/// assignment of mines/safes to hidden cells that simultaneously satisfies all
/// constraints.
///
/// # Two kinds of hidden cells
///
/// ```text
///  ┌───┬───┬───┐
///  │ 2 │ ? │ ? │  ← "border" cells (? adjacent to a numbered cell)
///  ├───┼───┼───┤     These are directly constrained.
///  │ ? │ ? │ ? │  ← "interior" cells (? not adjacent to any numbered cell)
///  └───┴───┴───┘     No constraint tells us exactly which are mines.
/// ```
///
/// For border cells we enumerate every valid assignment explicitly.
/// For interior cells we know only the total count of mines left over after
/// the border is fixed, so we treat them analytically (uniform distribution).
///
/// # Search algorithm
///
/// We process constraints one at a time (depth = constraint index).
/// At each level we look at the current constraint's unassigned neighbours and
/// pick which `needed` of the `m` unassigned cells are mines — that's C(m, needed)
/// choices.  We fix them, recurse to the next constraint, then backtrack.
///
/// Because earlier constraints already fixed some cells shared with later ones,
/// the branching factor shrinks rapidly → the tree is tiny compared with brute-force.
///
/// At each *leaf* (all constraints satisfied):
///   1. Count how many border mines were placed (`border_mines`).
///   2. Remaining mines = `mines_total - border_mines` must sit in interior cells.
///   3. There are C(n_interior, k_remaining) ways to do that — this is the leaf's *weight*.
///   4. Accumulate weighted mine counts for every cell.
///
/// Final probability for cell `c` = (sum of weights where c is a mine) / (total weight).
pub struct ConstraintSearch;

impl ConstraintSearch {
    pub fn new() -> Self {
        Self
    }

    pub fn calculate_with_progress(&self, game: &Minesweeper, tx: Sender<SimUpdate>) {
        // SimSetup::build extracts the list of hidden cells, constraints (numbered
        // cell → hidden neighbours + required mine count), certain mines/safes found
        // by constraint propagation, and the number of mines still to place.
        let Some(setup) = SimSetup::build(game) else {
            // No hidden cells reachable (game not started yet, already won/lost, …).
            let _ = tx.send(SimUpdate::Done {
                strategy: Strategy::ConstraintSearch,
                attempts: 0,
                valid: 0,
                memory_bytes: 0,
                probs: vec![vec![0.0; game.width]; game.height],
            });
            return;
        };

        // `n` is the number of hidden cells (indexed 0..n in `setup.hidden_cells`).
        let n = setup.hidden_cells.len();
        let memory_bytes = cs_memory_estimate(&setup);

        // Classify hidden cells into "border" (appear in at least one constraint)
        // and "interior" (not constrained at all).
        let constraint_cells: HashSet<usize> = setup
            .constraints
            .iter()
            .flat_map(|(neighbors, _)| neighbors.iter().copied())
            .collect();
        // Interior cell indices (into hidden_cells) — the ones with no numbered neighbour.
        let interior: Vec<usize> = (0..n).filter(|i| !constraint_cells.contains(i)).collect();

        // Send a progress snapshot to the GUI every 500 valid leaves so the user
        // sees partial results while the search is still running.
        let progress_every = 500usize;

        // These are declared outside the inner block so we can use them after
        // the `Dfs` struct is dropped (it holds a closure that borrows `tx`).
        let mine_counts;
        let total_weight;
        let valid_count;
        let step_count;

        {
            // Closure passed to Dfs::on_progress; called after every valid leaf.
            // Returns `false` to abort the search early (e.g. if the receiver hung up).
            let send_progress = |step: usize,
                                 valid: u32,
                                 counts: &[f64],
                                 total_wt: f64,
                                 interior_share: f64|
             -> bool {
                if step % progress_every != 0 {
                    return true; // Not a reporting step — keep going.
                }
                // Only materialise the interior cells when actually reporting;
                // doing it per leaf would cost more than the search itself.
                let counts = with_interior(counts, &interior, interior_share);
                let probs = build_probs(&counts, total_wt, &setup, game.width, game.height);
                tx.send(SimUpdate::Progress {
                    strategy: Strategy::ConstraintSearch,
                    attempts: step,
                    valid: valid as usize,
                    max_attempts: 0, // CS has no fixed budget — runs to completion.
                    memory_bytes,
                    probs,
                })
                .is_ok() // `false` if receiver dropped → abort.
            };

            let mut dfs = Dfs::new(
                n,
                &setup.constraints,
                &interior,
                setup.mines_to_place,
                send_progress,
            );
            // Start the DFS from constraint index 0 (root of the search tree).
            dfs.run(0);

            // Move results out before `dfs` (and its borrow of `tx`) is dropped.
            mine_counts = with_interior(&dfs.mine_counts, &interior, dfs.interior_weight);
            total_weight = dfs.total_weight;
            valid_count = dfs.valid_count;
            step_count = dfs.step_count;
        } // ← `dfs` (and the `send_progress` closure holding `&tx`) dropped here.

        // Send the final exact probabilities.
        let probs = build_probs(&mine_counts, total_weight, &setup, game.width, game.height);
        let _ = tx.send(SimUpdate::Done {
            strategy: Strategy::ConstraintSearch,
            attempts: step_count,
            valid: valid_count as usize,
            memory_bytes,
            probs,
        });
    }
}

impl ProbabilityStrategy for ConstraintSearch {
    /// Synchronous version used by the CLI / web — runs to completion and returns probs.
    fn calculate(&self, game: &Minesweeper) -> Vec<Vec<f64>> {
        let Some(setup) = SimSetup::build(game) else {
            return vec![vec![0.0; game.width]; game.height];
        };
        let n = setup.hidden_cells.len();
        let constraint_cells: HashSet<usize> = setup
            .constraints
            .iter()
            .flat_map(|(neighbors, _)| neighbors.iter().copied())
            .collect();
        let interior: Vec<usize> = (0..n).filter(|i| !constraint_cells.contains(i)).collect();

        // No progress callback needed — the caller blocks until done.
        let mut dfs = Dfs::new(
            n,
            &setup.constraints,
            &interior,
            setup.mines_to_place,
            |_, _, _, _, _| true,
        );
        dfs.run(0);

        let counts = with_interior(&dfs.mine_counts, &interior, dfs.interior_weight);
        build_probs(&counts, dfs.total_weight, &setup, game.width, game.height)
    }
}

// ---------------------------------------------------------------------------
// DFS engine
// ---------------------------------------------------------------------------

/// Depth-first search over the constraint tree.
///
/// `'a` ties the struct to the lifetime of the constraint/interior slices.
/// `F` is the progress callback type.
struct Dfs<'a, F> {
    /// Slice of (hidden-cell-indices-of-neighbours, required-mine-count) pairs,
    /// one entry per visible numbered cell.  This is the list of constraints we
    /// must satisfy, processed left-to-right (depth = index into this slice).
    constraints: &'a [(Vec<usize>, usize)],
    /// Indices (into hidden_cells) of cells not adjacent to any numbered cell.
    /// Their mine count is determined analytically at each leaf.
    interior: &'a [usize],
    /// Total mines that must be placed across ALL hidden cells.
    mines_total: usize,
    /// Current partial assignment.  `None` = not yet decided, `Some(true)` = mine,
    /// `Some(false)` = safe.  Indexed by hidden-cell index (0..n).
    assignment: Vec<Option<bool>>,
    /// Weighted mine-hit counter per hidden cell, accumulated across all valid leaves.
    /// `mine_counts[i]` = Σ weight over all leaves where cell i is a mine.
    /// Interior cells are *not* included here — see `interior_weight`.
    mine_counts: Vec<f64>,
    /// Number of border cells currently assigned as mines. Maintained as the
    /// search walks rather than recounted at each leaf, which was O(hidden) per
    /// leaf and dominated the whole search on a large board.
    border_mines: usize,
    /// The cells behind that count, newest last, so a leaf can credit exactly the
    /// cells it placed mines on instead of scanning every cell.
    mine_cells: Vec<usize>,
    /// Every interior cell gets the same contribution from a given leaf, so bank
    /// it once here and spread it over them when the numbers are read out.
    interior_weight: f64,
    /// `weight[k]` for placing k mines among the interior cells, precomputed
    /// because the interior never changes during a search.
    interior_ways: Vec<f64>,
    /// Sum of weights across all valid leaves.
    /// Dividing `mine_counts[i]` by this gives the exact mine probability for cell i.
    total_weight: f64,
    /// Number of valid leaves (constraints fully satisfied, mine counts feasible).
    valid_count: u32,
    /// Total leaves processed (valid + pruned-at-leaf level for mine-count check).
    step_count: usize,
    /// Called after each valid leaf with `(step, valid, border_counts,
    /// total_weight, interior_share)`; returns `false` to abort the search early.
    /// The counts exclude interior cells, which all share `interior_share` —
    /// materialising the full grid on every leaf would cost more than the search.
    on_progress: F,
    /// Set to `true` when `on_progress` returns `false`; causes all recursion to unwind.
    aborted: bool,
}

impl<'a, F> Dfs<'a, F>
where
    F: FnMut(usize, u32, &[f64], f64, f64) -> bool,
{
    fn new(
        n: usize,
        constraints: &'a [(Vec<usize>, usize)],
        interior: &'a [usize],
        mines_total: usize,
        on_progress: F,
    ) -> Self {
        Self {
            constraints,
            interior,
            mines_total,
            assignment: vec![None; n],       // all cells start undecided
            mine_counts: vec![0.0; n],
            border_mines: 0,
            mine_cells: Vec::new(),
            interior_weight: 0.0,
            interior_ways: scaled_binomials(interior.len()),
            total_weight: 0.0,
            valid_count: 0,
            step_count: 0,
            on_progress,
            aborted: false,
        }
    }

    /// Recursively assign mines/safes to satisfy `constraints[constraint_idx]`,
    /// then call `run(constraint_idx + 1)`.  Backtracks when done.
    fn run(&mut self, constraint_idx: usize) {
        if self.aborted {
            return;
        }

        // Base case: all constraints satisfied → score this leaf.
        if constraint_idx == self.constraints.len() {
            self.process_leaf();
            return;
        }

        // Clone the constraint data to avoid holding a borrow of `self.constraints`
        // while we mutate `self.assignment` during recursion.
        let (neighbors, required) = {
            let c = &self.constraints[constraint_idx];
            (c.0.clone(), c.1)
        };

        // How many of this constraint's neighbours are already marked as mines
        // by a previous constraint that shares cells with this one?
        let already_mines: usize = neighbors
            .iter()
            .filter(|&&i| self.assignment[i] == Some(true))
            .count();

        // Pruning: if we've already exceeded the required count, this branch is invalid.
        if already_mines > required {
            return;
        }

        // Cells in this constraint not yet assigned by an earlier constraint.
        let unassigned: Vec<usize> = neighbors
            .iter()
            .filter(|&&i| self.assignment[i].is_none())
            .copied()
            .collect();

        // How many additional mines we still need to place from the unassigned cells.
        let needed = required - already_mines;
        let m = unassigned.len();

        // Pruning: can't satisfy the constraint if there aren't enough free cells.
        if needed > m {
            return;
        }

        // Fast path: all unassigned cells are already decided (needed == 0 and m == 0,
        // or m > 0 but needed == 0 means they must all be safe).
        if m == 0 {
            // Every cell in this constraint is already fixed; constraint is satisfied.
            self.run(constraint_idx + 1);
            return;
        }

        if needed == 0 {
            // Zero additional mines needed → every unassigned neighbour must be safe.
            for &cell in &unassigned {
                self.set_cell(cell, false);
            }
            self.run(constraint_idx + 1);
            self.clear_cells(&unassigned);
            return;
        }

        if needed == m {
            // All unassigned neighbours must be mines (no choice).
            for &cell in &unassigned {
                self.set_cell(cell, true);
            }
            if self.border_mines <= self.mines_total {
                self.run(constraint_idx + 1);
            }
            self.clear_cells(&unassigned);
            return;
        }

        // General case: choose `needed` mines out of `m` unassigned cells.
        // We iterate over all C(m, needed) subsets in lexicographic order.
        //
        // `combo` holds the *positions* (0..m) of the chosen mines.
        // Initially [0, 1, 2, …, needed-1] — the first subset.
        let mut combo: Vec<usize> = (0..needed).collect();
        // Allocated once for the whole loop rather than per combination.
        let mut is_mine_pos = vec![false; m];
        loop {
            if self.aborted {
                break;
            }

            // Apply this combination: mark selected positions as mines, rest as safe.
            is_mine_pos.iter_mut().for_each(|slot| *slot = false);
            for &ci in &combo {
                is_mine_pos[ci] = true;
            }
            for (j, &cell) in unassigned.iter().enumerate() {
                self.set_cell(cell, is_mine_pos[j]);
            }

            // Global budget: a branch that has already placed more mines than the
            // board holds cannot lead anywhere. Without this the search descends
            // through every remaining constraint before noticing at the leaf.
            if self.border_mines <= self.mines_total {
                self.run(constraint_idx + 1);
            }

            // Backtrack: clear all assignments made by this constraint level so
            // the next combination starts from a clean slate.
            self.clear_cells(&unassigned);

            // Advance `combo` to the next combination in lexicographic order.
            // Find the rightmost position that can still be incremented.
            //
            // Example with m=5, needed=3, combo=[1,3,4]:
            //   i starts at 3 (== needed).
            //   combo[2]=4 == 5-3+2=4 → at max, decrement i to 2.
            //   combo[1]=3 == 5-3+1=3 → at max, decrement i to 1.
            //   combo[0]=1 != 5-3+0=2 → stop, i=1.
            //   Increment combo[0] to 2, fill rest: combo = [2,3,4].
            let mut i = needed;
            while i > 0 && combo[i - 1] == m - needed + i - 1 {
                i -= 1;
            }
            if i == 0 {
                break; // All combinations exhausted.
            }
            combo[i - 1] += 1;
            for j in i..needed {
                combo[j] = combo[j - 1] + 1;
            }
        }
    }

    /// Record one assignment, keeping the running mine count in step.
    fn set_cell(&mut self, cell: usize, mine: bool) {
        self.assignment[cell] = Some(mine);
        if mine {
            self.border_mines += 1;
            self.mine_cells.push(cell);
        }
    }

    /// Undo the assignments made for one constraint level.
    ///
    /// Cleared in reverse so `mine_cells` unwinds exactly as it was built — it is
    /// a stack, and the leaf handler depends on it holding precisely the cells
    /// currently assigned as mines.
    fn clear_cells(&mut self, cells: &[usize]) {
        for &cell in cells.iter().rev() {
            if self.assignment[cell] == Some(true) {
                self.border_mines -= 1;
                debug_assert_eq!(self.mine_cells.last(), Some(&cell));
                self.mine_cells.pop();
            }
            self.assignment[cell] = None;
        }
    }

    /// Called when all constraints are satisfied (we're at a leaf of the search tree).
    ///
    /// Checks global mine-count feasibility, computes the leaf weight, and
    /// accumulates mine probability contributions.
    ///
    /// This runs once per valid layout — millions of times on a dense board — so
    /// everything here is O(mines placed) and nothing is O(board).
    fn process_leaf(&mut self) {
        // How many mines remain for interior (unconstrained) cells?
        let k_i = match self.mines_total.checked_sub(self.border_mines) {
            Some(v) => v,
            None => return, // More border mines than the total — impossible layout.
        };

        // Can't place k_i mines in n_i cells if k_i > n_i.
        let n_i = self.interior.len();
        if k_i > n_i {
            return;
        }

        // The leaf's weight = number of ways to arrange the remaining k_i mines
        // among the n_i interior cells.  C(n_i, k_i) boards all look like this
        // border assignment but differ in which interior cells are mines.
        let weight = self.interior_ways[k_i];
        self.total_weight += weight;
        self.valid_count += 1;

        // For each border cell assigned as a mine, add `weight` to its count.
        // (All C(n_i,k_i) boards that share this border pattern have this mine here.)
        for i in 0..self.mine_cells.len() {
            let cell = self.mine_cells[i];
            self.mine_counts[cell] += weight;
        }

        // Interior cells: each one is a mine in k_i/n_i fraction of the C(n_i,k_i)
        // boards, so every interior cell earns the same amount from this leaf.
        // Bank the per-cell share instead of writing it to all of them now.
        if n_i > 0 && k_i > 0 {
            self.interior_weight += weight * k_i as f64 / n_i as f64;
        }

        self.step_count += 1;
        // Notify the caller; if it returns false the search is aborted.
        if !(self.on_progress)(
            self.step_count,
            self.valid_count,
            &self.mine_counts,
            self.total_weight,
            self.interior_weight,
        ) {
            self.aborted = true;
        }
    }
}

/// Spread the banked interior share back over the interior cells, producing the
/// full per-cell counts the probability builder expects.
fn with_interior(border_counts: &[f64], interior: &[usize], share: f64) -> Vec<f64> {
    let mut counts = border_counts.to_vec();
    for &i in interior {
        counts[i] += share;
    }
    counts
}

/// Weights `C(n, k)` for every k in `0..=n`, scaled so the largest is 1.
///
/// Scaled because the true values overflow: `C(700, 200)` is around 10^200 and a
/// bigger board saturates `f64` to infinity, which turns the final division into
/// NaN. Only ratios between these weights are ever used, so dividing the table by
/// its largest entry costs nothing and keeps every value finite.
fn scaled_binomials(n: usize) -> Vec<f64> {
    let mut ln_factorial = vec![0.0f64; n + 1];
    for i in 1..=n {
        ln_factorial[i] = ln_factorial[i - 1] + (i as f64).ln();
    }
    let ln_weights: Vec<f64> = (0..=n)
        .map(|k| ln_factorial[n] - ln_factorial[k] - ln_factorial[n - k])
        .collect();
    let peak = ln_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ln_weights.iter().map(|&w| (w - peak).exp()).collect()
}

// ---------------------------------------------------------------------------
// Memory estimation
// ---------------------------------------------------------------------------

/// Rough heap estimate for the Constraint Search strategy's working set.
///
/// Accounts for SimSetup storage (shared with MC) plus the DFS-specific
/// structures: assignment vector, mine-count accumulator, and call-stack
/// frame cost (combo + unassigned + is_mine_pos vectors per depth level).
fn cs_memory_estimate(setup: &SimSetup) -> usize {
    let n = setup.hidden_cells.len();
    let total_neighbors: usize = setup.constraints.iter().map(|(ns, _)| ns.len()).sum();
    let c = setup.constraints.len();

    // SimSetup heap (same formula as in mc_memory_estimate)
    let setup_heap = mc_memory_estimate(setup);

    // DFS working set: the call stack goes `c` levels deep (one per constraint).
    // At each level we allocate three temporary vectors of size ≈ avg_unassigned.
    let avg_unassigned = total_neighbors.checked_div(c).unwrap_or(0);
    let stack_frame = avg_unassigned * 8  // combo: Vec<usize>
        + avg_unassigned * 8             // unassigned: Vec<usize>
        + avg_unassigned                 // is_mine_pos: Vec<bool>
        + 64;                            // Dfs struct overhead per frame
    let dfs_stack = c * stack_frame;

    let working = n          // assignment: Vec<Option<bool>> (1 byte each)
        + n * 8              // mine_counts: Vec<f64>
        + dfs_stack;

    setup_heap + working
}