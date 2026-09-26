//! Turning a board into the constraint problem the solver works on.
//!
//! Every estimator starts here: the unopened cells become the unknowns, every
//! visible number becomes "exactly k of these are mines", and [`propagate`]
//! settles whatever local rules alone can settle before any search begins.
//!
//! This was `monte_carlo.rs` until the sampler was deleted. Sampling cannot
//! produce a correct answer, and measured against the decomposing exact search
//! it could not even produce one faster: over 71 mid-game positions the exact
//! search answered all 71 in 20 ms in total, while Monte Carlo answered 20 of
//! them and took 4.4 seconds doing it — and among those answers was a cell it
//! called 0%, which every front-end reads as proof of safety, that was not safe.

use std::collections::HashSet;

use crate::{CellContent, CellState, Minesweeper};

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
    /// Cells proven to be safe by constraint propagation (always probability 0).
    pub(crate) certain_safe: Vec<(usize, usize)>,
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

        // Reverse lookup, board cell -> index into `all_hidden`. The neighbour
        // scan below used to linear-search `all_hidden` for every neighbour of
        // every visible cell, which is O(visible x hidden) and grows with board
        // area — costly here because auto-reveal calls this repeatedly.
        let mut hidden_index = vec![usize::MAX; game.width * game.height];
        for (i, &(hx, hy)) in all_hidden.iter().enumerate() {
            hidden_index[hy * game.width + hx] = i;
        }

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
                // A zero counts too: "none of my neighbours is a mine" is just as
                // strong a constraint as any other number. A cascade normally
                // leaves a 0 with no hidden neighbours, but a board restored from
                // a snapshot carries no such guarantee, so don't rely on it.
                let total = match game.grid[y][x].content {
                    CellContent::Empty(n) => n as usize,
                    CellContent::Mine => return None,
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
                            let idx = hidden_index[ny * game.width + nx];
                            if idx != usize::MAX {
                                hidden_neighbor_indices.push(idx);
                            }
                        }
                    }
                }

                let required = total.saturating_sub(visible_mine_neighbors);
                // A cell with nothing hidden around it constrains nothing.
                if hidden_neighbor_indices.is_empty() || required > hidden_neighbor_indices.len() {
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
        // Sets, not Vec::contains: propagation can determine most of the board,
        // and the linear scans below are then quadratic in the hidden count.
        let determined: HashSet<usize> = certain_mine_idxs
            .iter()
            .chain(certain_safe_idxs.iter())
            .copied()
            .collect();
        let uncertain: Vec<usize> = (0..n).filter(|i| !determined.contains(i)).collect();

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
                    .filter(|i| !determined.contains(i))
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
        let certain_safe: Vec<(usize, usize)> =
            certain_safe_idxs.iter().map(|&i| all_hidden[i]).collect();

        Some(Self {
            hidden_cells,
            constraints,
            mines_to_place,
            certain_mines,
            certain_safe,
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

/// Rough heap estimate for a solve's working set, for the memory readouts.
pub(crate) fn memory_estimate(setup: &SimSetup) -> usize {
    let n = setup.hidden_cells.len();
    let total_neighbors: usize = setup.constraints.iter().map(|(ns, _)| ns.len()).sum();
    let c = setup.constraints.len();

    // SimSetup heap
    let setup_heap = n * 16               // hidden_cells: Vec<(usize, usize)>
        + total_neighbors * 8 + c * 24   // constraints: Vec<(Vec<usize>, usize)>
        + setup.certain_mines.len() * 16; // certain_mines

    // Working set: mine_counts + border/interior vecs + b_indices + is_mine_b
    //              + border_constraints + border_pos HashMap (rough 48 B/entry)
    let working = n * 8      // mine_counts: Vec<f64>
        + n * 8              // border + interior vecs (upper bound n each)
        + n * 8              // b_indices
        + n                  // is_mine_b: Vec<bool>
        + total_neighbors * 8 + c * 24  // border_constraints (same size as constraints)
        + n * 48;            // border_pos HashMap entries

    setup_heap + working
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
