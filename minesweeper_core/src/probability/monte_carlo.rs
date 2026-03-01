use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::Sender;

use crate::{CellContent, CellState, Minesweeper};

use super::{ProbabilityStrategy, SimUpdate, Strategy};

/// Estimates mine probabilities using Monte Carlo sampling, or exact
/// enumeration when the total number of possible layouts is small enough.
///
/// The sampling path separates *border* cells (adjacent to revealed numbers,
/// involved in constraints) from *interior* cells (completely unconstrained).
/// Only border cells are sampled against constraints; interior cell probabilities
/// are derived analytically from the remaining mine count, increasing the
/// valid-sample rate by orders of magnitude early in the game.
pub struct MonteCarlo {
    pub max_valid: usize,
    pub max_attempts: usize,
}

impl MonteCarlo {
    pub fn new() -> Self {
        Self {
            max_valid: 10_000,
            max_attempts: 1_000_000,
        }
    }

    /// Like [`ProbabilityStrategy::calculate`] but sends periodic [`SimUpdate`]s
    /// through `tx` so callers can display live progress.
    pub fn calculate_with_progress(&self, game: &Minesweeper, tx: Sender<SimUpdate>) {
        let Some(setup) = SimSetup::build(game) else {
            let _ = tx.send(SimUpdate::Done {
                strategy: Strategy::MonteCarlo,
                attempts: 0,
                valid: 0,
                probs: vec![vec![0.0; game.width]; game.height],
            });
            return;
        };

        let n = setup.hidden_cells.len();
        let total_combos = combinations(n, setup.mines_to_place);
        let use_exact = total_combos <= self.max_valid as f64;
        let max_steps = if use_exact { total_combos as usize } else { self.max_attempts };
        let progress_every = usize::max(1, max_steps / 20);

        let send_progress = |step: usize, valid: u32, counts: &[f64], total_wt: f64| -> bool {
            let probs = build_probs(counts, total_wt, &setup, game.width, game.height);
            tx.send(SimUpdate::Progress {
                strategy: Strategy::MonteCarlo,
                attempts: step,
                valid: valid as usize,
                max_attempts: max_steps,
                probs,
            })
            .is_ok()
        };

        let (mine_counts, total_weight, valid_count, attempts) = if use_exact {
            self.enumerate_all(&setup, |step, valid, counts, wt| {
                if step % progress_every == 0 && step > 0 {
                    send_progress(step, valid, counts, wt)
                } else {
                    true
                }
            })
        } else {
            self.run_loop(&setup, |attempt, valid, counts, wt| {
                if attempt % progress_every == 0 && attempt > 0 {
                    send_progress(attempt, valid, counts, wt)
                } else {
                    true
                }
            })
        };

        let probs = build_probs(&mine_counts, total_weight, &setup, game.width, game.height);
        let _ = tx.send(SimUpdate::Done {
            strategy: Strategy::MonteCarlo,
            attempts,
            valid: valid_count as usize,
            probs,
        });
    }

    /// Iterate every possible mine layout exhaustively (exact mode).
    fn enumerate_all<F>(&self, setup: &SimSetup, mut on_progress: F) -> (Vec<f64>, f64, u32, usize)
    where
        F: FnMut(usize, u32, &[f64], f64) -> bool,
    {
        let n = setup.hidden_cells.len();
        let k = setup.mines_to_place;
        let mut mine_counts = vec![0.0f64; n];
        let mut valid_count = 0u32;
        let mut total = 0usize;

        if k == 0 || n == 0 {
            return (mine_counts, 0.0, 0, 0);
        }

        let mut combo: Vec<usize> = (0..k).collect();

        loop {
            let mut is_mine = vec![false; n];
            for &idx in &combo {
                is_mine[idx] = true;
            }

            let valid = setup.constraints.iter().all(|(neighbors, required)| {
                neighbors.iter().filter(|&&i| is_mine[i]).count() == *required
            });

            if valid {
                valid_count += 1;
                for &idx in &combo {
                    mine_counts[idx] += 1.0;
                }
            }

            total += 1;

            if !on_progress(total, valid_count, &mine_counts, valid_count as f64) {
                break;
            }

            // Advance to the next combination in lexicographic order.
            let mut i = k;
            while i > 0 && combo[i - 1] == n - k + i - 1 {
                i -= 1;
            }
            if i == 0 {
                break;
            }
            combo[i - 1] += 1;
            for j in i..k {
                combo[j] = combo[j - 1] + 1;
            }
        }

        (mine_counts, valid_count as f64, valid_count, total)
    }

    /// Border-interior separated Monte Carlo sampling.
    ///
    /// Constraints only reference *border* cells — cells adjacent to a revealed
    /// number. *Interior* cells (unconstrained) can hold any of the remaining
    /// mines uniformly. We sample `k_b` mines from border cells only, check
    /// constraints, and weight each valid sample by `C(n_interior, k - k_b)`.
    /// Interior cell probabilities are derived analytically per sample.
    ///
    /// This gives an unbiased estimator while achieving a dramatically higher
    /// valid-sample rate compared to sampling across all cells.
    fn run_loop<F>(&self, setup: &SimSetup, mut on_progress: F) -> (Vec<f64>, f64, u32, usize)
    where
        F: FnMut(usize, u32, &[f64], f64) -> bool,
    {
        let n = setup.hidden_cells.len();
        let k = setup.mines_to_place;

        // --- Border / interior split ---
        let border_set: HashSet<usize> = setup
            .constraints
            .iter()
            .flat_map(|(neighbors, _)| neighbors.iter().copied())
            .collect();
        let border: Vec<usize> = (0..n).filter(|i| border_set.contains(i)).collect();
        let interior: Vec<usize> = (0..n).filter(|i| !border_set.contains(i)).collect();

        let b = border.len();
        let n_int = interior.len();

        // Re-index constraints to reference positions within `border`.
        let border_pos: HashMap<usize, usize> =
            border.iter().enumerate().map(|(bi, &orig)| (orig, bi)).collect();
        let border_constraints: Vec<(Vec<usize>, usize)> = setup
            .constraints
            .iter()
            .map(|(neighbors, req)| {
                (
                    neighbors.iter().map(|&orig| border_pos[&orig]).collect(),
                    *req,
                )
            })
            .collect();

        let k_b_min = k.saturating_sub(n_int);
        let k_b_max = k.min(b);
        if k_b_min > k_b_max {
            return (vec![0.0; n], 0.0, 0, 0);
        }
        let k_b_range = k_b_max - k_b_min + 1;

        let mut mine_counts = vec![0.0f64; n];
        let mut total_weight = 0.0f64;
        let mut valid_count = 0u32;
        let mut rng = rand::thread_rng();
        let mut b_indices: Vec<usize> = (0..b).collect();
        let mut attempt = 0;

        for _ in 0..self.max_attempts {
            if valid_count >= self.max_valid as u32 {
                break;
            }

            // Uniformly choose how many mines go on border cells.
            let k_b = if k_b_range == 1 {
                k_b_min
            } else {
                rng.gen_range(k_b_min..=k_b_max)
            };
            let k_i = k - k_b;

            // Partial Fisher-Yates on border cells to sample k_b mines.
            for i in 0..k_b {
                let j = rng.gen_range(i..b);
                b_indices.swap(i, j);
            }

            let mut is_mine_b = vec![false; b];
            for &idx in &b_indices[..k_b] {
                is_mine_b[idx] = true;
            }

            // Validate constraints (all defined over border cells).
            let valid = border_constraints.iter().all(|(neighbors, required)| {
                neighbors.iter().filter(|&&i| is_mine_b[i]).count() == *required
            });

            if valid {
                // Weight = C(n_int, k_i): number of interior completions for this border config.
                // Dividing by C(b, k_b) * (1/k_b_range) cancels out in the ratio, leaving
                // C(n_int, k_i) as the effective importance weight.
                let weight = combinations(n_int, k_i);
                total_weight += weight;
                valid_count += 1;

                for (bi, &orig) in border.iter().enumerate() {
                    if is_mine_b[bi] {
                        mine_counts[orig] += weight;
                    }
                }

                // Each interior cell is equally likely to hold a mine: k_i / n_int per sample.
                if n_int > 0 && k_i > 0 {
                    let frac = weight * k_i as f64 / n_int as f64;
                    for &orig in &interior {
                        mine_counts[orig] += frac;
                    }
                }
            }

            attempt += 1;
            if !on_progress(attempt, valid_count, &mine_counts, total_weight) {
                break;
            }
        }

        (mine_counts, total_weight, valid_count, attempt)
    }
}

impl ProbabilityStrategy for MonteCarlo {
    fn calculate(&self, game: &Minesweeper) -> Vec<Vec<f64>> {
        let Some(setup) = SimSetup::build(game) else {
            return vec![vec![0.0; game.width]; game.height];
        };
        let n = setup.hidden_cells.len();
        let use_exact = combinations(n, setup.mines_to_place) <= self.max_valid as f64;
        let (mine_counts, total_weight, _, _) = if use_exact {
            self.enumerate_all(&setup, |_, _, _, _| true)
        } else {
            self.run_loop(&setup, |_, _, _, _| true)
        };
        build_probs(&mine_counts, total_weight, &setup, game.width, game.height)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub(crate) struct SimSetup {
    /// Uncertain hidden cells — those not yet determined by propagation.
    pub(crate) hidden_cells: Vec<(usize, usize)>,
    /// Constraints re-indexed to reference `hidden_cells`.
    pub(crate) constraints: Vec<(Vec<usize>, usize)>,
    /// Remaining mines to distribute among `hidden_cells`.
    pub(crate) mines_to_place: usize,
    /// Cells proven to be mines by constraint propagation (always probability 1).
    pub(crate) certain_mines: Vec<(usize, usize)>,
}

impl SimSetup {
    pub(crate) fn build(game: &Minesweeper) -> Option<Self> {
        let all_hidden: Vec<(usize, usize)> = (0..game.height)
            .flat_map(|y| (0..game.width).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                matches!(game.grid[y][x].state, CellState::Hidden | CellState::Flagged)
            })
            .collect();

        let n = all_hidden.len();

        let visible_mine_count = (0..game.height)
            .flat_map(|y| (0..game.width).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                matches!(game.grid[y][x].content, CellContent::Mine)
                    && game.grid[y][x].state == CellState::Visible
            })
            .count();

        let mines_to_place = game.mines_count.saturating_sub(visible_mine_count);

        if mines_to_place > n {
            return None;
        }

        let raw_constraints: Vec<(Vec<usize>, usize)> = (0..game.height)
            .flat_map(|y| (0..game.width).map(move |x| (x, y)))
            .filter_map(|(x, y)| {
                if game.grid[y][x].state != CellState::Visible {
                    return None;
                }
                let total = match game.grid[y][x].content {
                    CellContent::Empty(n) if n > 0 => n as usize,
                    _ => return None,
                };

                let mut visible_mine_neighbors = 0usize;
                let mut hidden_neighbor_indices = Vec::new();

                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;
                        if nx < 0
                            || nx >= game.width as isize
                            || ny < 0
                            || ny >= game.height as isize
                        {
                            continue;
                        }
                        let nx = nx as usize;
                        let ny = ny as usize;
                        if matches!(game.grid[ny][nx].content, CellContent::Mine)
                            && game.grid[ny][nx].state == CellState::Visible
                        {
                            visible_mine_neighbors += 1;
                        } else if matches!(
                            game.grid[ny][nx].state,
                            CellState::Hidden | CellState::Flagged
                        ) {
                            if let Some(idx) =
                                all_hidden.iter().position(|&(hx, hy)| hx == nx && hy == ny)
                            {
                                hidden_neighbor_indices.push(idx);
                            }
                        }
                    }
                }

                let required = total.saturating_sub(visible_mine_neighbors);
                if required > hidden_neighbor_indices.len() {
                    None
                } else {
                    Some((hidden_neighbor_indices, required))
                }
            })
            .collect();

        // Run constraint propagation to determine certain mines / safe cells.
        let (certain_mine_idxs, certain_safe_idxs, constraints, mines_to_place) =
            propagate(n, raw_constraints, mines_to_place);

        // Compact hidden_cells to only uncertain cells, re-indexing constraints.
        let uncertain: Vec<usize> = (0..n)
            .filter(|i| !certain_mine_idxs.contains(i) && !certain_safe_idxs.contains(i))
            .collect();

        let mut old_to_new = vec![usize::MAX; n];
        for (new, &old) in uncertain.iter().enumerate() {
            old_to_new[old] = new;
        }

        let hidden_cells: Vec<(usize, usize)> = uncertain.iter().map(|&i| all_hidden[i]).collect();

        let constraints: Vec<(Vec<usize>, usize)> = constraints
            .into_iter()
            .filter_map(|(neighbors, required)| {
                let new_neighbors: Vec<usize> = neighbors
                    .into_iter()
                    .filter(|&i| !certain_mine_idxs.contains(&i) && !certain_safe_idxs.contains(&i))
                    .map(|i| old_to_new[i])
                    .collect();
                if new_neighbors.is_empty() {
                    None
                } else {
                    Some((new_neighbors, required))
                }
            })
            .collect();

        let certain_mines: Vec<(usize, usize)> =
            certain_mine_idxs.iter().map(|&i| all_hidden[i]).collect();

        Some(Self {
            hidden_cells,
            constraints,
            mines_to_place,
            certain_mines,
        })
    }
}

/// Constraint propagation to fixpoint.
///
/// Determines cells that are certainly mines or certainly safe:
/// - `required == 0`                   → all undetermined neighbors are safe
/// - `required == undetermined.len()`  → all undetermined neighbors are mines
/// - `mines_to_place == 0`             → all remaining uncertain cells are safe
/// - `mines_to_place == n_uncertain`   → all remaining uncertain cells are mines
///
/// Returns `(certain_mine_indices, certain_safe_indices, updated_constraints, remaining_mines)`.
fn propagate(
    n: usize,
    mut constraints: Vec<(Vec<usize>, usize)>,
    mut mines_to_place: usize,
) -> (Vec<usize>, Vec<usize>, Vec<(Vec<usize>, usize)>, usize) {
    let mut is_mine = vec![false; n];
    let mut is_safe = vec![false; n];

    loop {
        let mut changed = false;

        for (neighbors, required) in &constraints {
            let known_mines: usize = neighbors.iter().filter(|&&i| is_mine[i]).count();
            let undetermined: Vec<usize> = neighbors
                .iter()
                .filter(|&&i| !is_mine[i] && !is_safe[i])
                .copied()
                .collect();
            let remaining = (*required).saturating_sub(known_mines);

            if remaining == 0 {
                for &i in &undetermined {
                    if !is_safe[i] {
                        is_safe[i] = true;
                        changed = true;
                    }
                }
            } else if remaining == undetermined.len() {
                for &i in &undetermined {
                    if !is_mine[i] {
                        is_mine[i] = true;
                        changed = true;
                    }
                }
            }
        }

        // Global constraint: total mines must equal mines_to_place.
        let confirmed_mines = is_mine.iter().filter(|&&m| m).count();
        let uncertain_count = (0..n).filter(|&i| !is_mine[i] && !is_safe[i]).count();
        let remaining_global = mines_to_place.saturating_sub(confirmed_mines);

        if remaining_global == 0 {
            for i in 0..n {
                if !is_mine[i] && !is_safe[i] {
                    is_safe[i] = true;
                    changed = true;
                }
            }
        } else if remaining_global == uncertain_count && remaining_global > 0 {
            for i in 0..n {
                if !is_mine[i] && !is_safe[i] {
                    is_mine[i] = true;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    let certain_mines: Vec<usize> = (0..n).filter(|&i| is_mine[i]).collect();
    let certain_safe: Vec<usize> = (0..n).filter(|&i| is_safe[i]).collect();
    mines_to_place = mines_to_place.saturating_sub(certain_mines.len());

    // Update constraints: subtract known mines from required counts.
    let updated_constraints: Vec<(Vec<usize>, usize)> = constraints
        .drain(..)
        .map(|(neighbors, required)| {
            let known_mines_here = neighbors.iter().filter(|&&i| is_mine[i]).count();
            (neighbors, required.saturating_sub(known_mines_here))
        })
        .collect();

    (certain_mines, certain_safe, updated_constraints, mines_to_place)
}

pub(crate) fn build_probs(
    mine_counts: &[f64],
    total_weight: f64,
    setup: &SimSetup,
    width: usize,
    height: usize,
) -> Vec<Vec<f64>> {
    let mut probs = vec![vec![0.0f64; width]; height];
    if total_weight > 0.0 {
        for (idx, &(x, y)) in setup.hidden_cells.iter().enumerate() {
            probs[y][x] = (mine_counts[idx] / total_weight).clamp(0.0, 1.0);
        }
    }
    // Cells proven to be mines by propagation are always 1.0.
    for &(x, y) in &setup.certain_mines {
        probs[y][x] = 1.0;
    }
    probs
}

/// C(n, k) as f64. Uses the multiplicative formula to avoid integer overflow.
pub fn combinations(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    (0..k).fold(1.0_f64, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}
