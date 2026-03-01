use std::collections::HashSet;
use std::sync::mpsc::Sender;

use crate::Minesweeper;

use super::{ProbabilityStrategy, SimUpdate, Strategy};
use super::monte_carlo::{SimSetup, build_probs, combinations};

/// Exact mine probability estimation using depth-first constraint enumeration.
///
/// Processes each numbered-cell constraint in turn. For each constraint,
/// enumerates all valid mine/safe assignments for its unresolved neighbors,
/// then recurses to the next constraint. At each leaf (all constraints
/// satisfied), distributes remaining mines among unconstrained "interior"
/// cells uniformly, weighting the leaf by `C(n_interior, k_remaining)`.
///
/// This is exact: every valid mine layout is counted with its correct
/// relative weight. It is typically far faster than exhaustive enumeration
/// of all `C(n, k)` combinations because the constraint structure prunes
/// the search tree heavily.
pub struct ConstraintSearch;

impl ConstraintSearch {
    pub fn new() -> Self {
        Self
    }

    pub fn calculate_with_progress(&self, game: &Minesweeper, tx: Sender<SimUpdate>) {
        let Some(setup) = SimSetup::build(game) else {
            let _ = tx.send(SimUpdate::Done {
                strategy: Strategy::ConstraintSearch,
                attempts: 0,
                valid: 0,
                probs: vec![vec![0.0; game.width]; game.height],
            });
            return;
        };

        let n = setup.hidden_cells.len();
        let constraint_cells: HashSet<usize> = setup
            .constraints
            .iter()
            .flat_map(|(neighbors, _)| neighbors.iter().copied())
            .collect();
        let interior: Vec<usize> = (0..n).filter(|i| !constraint_cells.contains(i)).collect();

        // Report every 500 valid leaves.
        let progress_every = 500usize;

        let mine_counts;
        let total_weight;
        let valid_count;
        let step_count;

        {
            let send_progress = |step: usize, valid: u32, counts: &[f64], total_wt: f64| -> bool {
                if step % progress_every != 0 {
                    return true;
                }
                let probs = build_probs(counts, total_wt, &setup, game.width, game.height);
                tx.send(SimUpdate::Progress {
                    strategy: Strategy::ConstraintSearch,
                    attempts: step,
                    valid: valid as usize,
                    max_attempts: 0,
                    probs,
                })
                .is_ok()
            };

            let mut dfs = Dfs::new(
                n,
                &setup.constraints,
                &interior,
                setup.mines_to_place,
                send_progress,
            );
            dfs.run(0);

            mine_counts = dfs.mine_counts;
            total_weight = dfs.total_weight;
            valid_count = dfs.valid_count;
            step_count = dfs.step_count;
        }

        let probs = build_probs(&mine_counts, total_weight, &setup, game.width, game.height);
        let _ = tx.send(SimUpdate::Done {
            strategy: Strategy::ConstraintSearch,
            attempts: step_count,
            valid: valid_count as usize,
            probs,
        });
    }
}

impl ProbabilityStrategy for ConstraintSearch {
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

        let mut dfs = Dfs::new(
            n,
            &setup.constraints,
            &interior,
            setup.mines_to_place,
            |_, _, _, _| true,
        );
        dfs.run(0);

        build_probs(&dfs.mine_counts, dfs.total_weight, &setup, game.width, game.height)
    }
}

// ---------------------------------------------------------------------------
// DFS engine
// ---------------------------------------------------------------------------

struct Dfs<'a, F> {
    constraints: &'a [(Vec<usize>, usize)],
    interior: &'a [usize],
    mines_total: usize,
    /// Per-cell assignment: None = unvisited, Some(true) = mine, Some(false) = safe.
    assignment: Vec<Option<bool>>,
    mine_counts: Vec<f64>,
    total_weight: f64,
    valid_count: u32,
    /// Total leaf nodes processed (valid + invalid).
    step_count: usize,
    on_progress: F,
    aborted: bool,
}

impl<'a, F> Dfs<'a, F>
where
    F: FnMut(usize, u32, &[f64], f64) -> bool,
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
            assignment: vec![None; n],
            mine_counts: vec![0.0; n],
            total_weight: 0.0,
            valid_count: 0,
            step_count: 0,
            on_progress,
            aborted: false,
        }
    }

    fn run(&mut self, constraint_idx: usize) {
        if self.aborted {
            return;
        }

        if constraint_idx == self.constraints.len() {
            self.process_leaf();
            return;
        }

        // Clone to avoid holding a borrow of `self` while recursing.
        let (neighbors, required) = {
            let c = &self.constraints[constraint_idx];
            (c.0.clone(), c.1)
        };

        let already_mines: usize = neighbors
            .iter()
            .filter(|&&i| self.assignment[i] == Some(true))
            .count();

        if already_mines > required {
            return; // Constraint already violated.
        }

        let unassigned: Vec<usize> = neighbors
            .iter()
            .filter(|&&i| self.assignment[i].is_none())
            .copied()
            .collect();

        let needed = required - already_mines;
        let m = unassigned.len();

        if needed > m {
            return; // Not enough cells left to satisfy the constraint.
        }

        if m == 0 {
            // All cells already determined; just recurse.
            self.run(constraint_idx + 1);
            return;
        }

        if needed == 0 {
            // All unassigned must be safe.
            for &cell in &unassigned {
                self.assignment[cell] = Some(false);
            }
            self.run(constraint_idx + 1);
            for &cell in &unassigned {
                self.assignment[cell] = None;
            }
            return;
        }

        if needed == m {
            // All unassigned must be mines.
            for &cell in &unassigned {
                self.assignment[cell] = Some(true);
            }
            self.run(constraint_idx + 1);
            for &cell in &unassigned {
                self.assignment[cell] = None;
            }
            return;
        }

        // General case: enumerate C(m, needed) subsets via lexicographic iteration.
        let mut combo: Vec<usize> = (0..needed).collect();
        loop {
            if self.aborted {
                break;
            }

            // Assign mines for this combo, safe for the rest.
            let mut is_mine_pos = vec![false; m];
            for &ci in &combo {
                is_mine_pos[ci] = true;
            }
            for (j, &cell) in unassigned.iter().enumerate() {
                self.assignment[cell] = Some(is_mine_pos[j]);
            }

            self.run(constraint_idx + 1);

            // Undo assignments.
            for &cell in &unassigned {
                self.assignment[cell] = None;
            }

            // Advance to the next combination in lexicographic order.
            let mut i = needed;
            while i > 0 && combo[i - 1] == m - needed + i - 1 {
                i -= 1;
            }
            if i == 0 {
                break;
            }
            combo[i - 1] += 1;
            for j in i..needed {
                combo[j] = combo[j - 1] + 1;
            }
        }
    }

    fn process_leaf(&mut self) {
        // Count how many border (constraint) cells are assigned as mines.
        let border_mines: usize = self.assignment.iter().filter(|a| **a == Some(true)).count();

        let k_i = match self.mines_total.checked_sub(border_mines) {
            Some(v) => v,
            None => return, // Impossible: more border mines than total.
        };
        let n_i = self.interior.len();

        if k_i > n_i {
            return; // Impossible: more interior mines needed than interior cells.
        }

        // Weight = number of ways to distribute k_i mines among n_i interior cells.
        let weight = combinations(n_i, k_i);
        self.total_weight += weight;
        self.valid_count += 1;

        // Accumulate border mine counts.
        for (i, a) in self.assignment.iter().enumerate() {
            if *a == Some(true) {
                self.mine_counts[i] += weight;
            }
        }

        // Each interior cell has uniform probability k_i / n_i per leaf.
        if n_i > 0 && k_i > 0 {
            let frac = weight * k_i as f64 / n_i as f64;
            for &i in self.interior {
                self.mine_counts[i] += frac;
            }
        }

        self.step_count += 1;
        if !(self.on_progress)(
            self.step_count,
            self.valid_count,
            &self.mine_counts,
            self.total_weight,
        ) {
            self.aborted = true;
        }
    }
}
