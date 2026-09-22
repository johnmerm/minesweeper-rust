//! Correctness gate for the mine-probability estimators.
//!
//! The exact strategies are checked against a brute-force oracle: on a small
//! board we enumerate *every* mine layout consistent with what the player can
//! see and count how often each cell is a mine. That is the definition of the
//! number the estimators are supposed to produce, so any disagreement is a bug
//! in the estimator, not in the test.
//!
//! Boards are built explicitly rather than through `Minesweeper::reveal`'s lazy
//! generation, so every position here is reproducible without an RNG.

use minesweeper_core::probability::{ConstraintSearch, ProbabilityStrategy};
use minesweeper_core::{CellContent, CellState, Minesweeper};

/// Build a board with mines at the given coordinates and all numbers filled in,
/// as if the player had already made their first click.
fn board(width: usize, height: usize, mines: &[(usize, usize)]) -> Minesweeper {
    let mut game = Minesweeper::new(width, height, mines.len());
    for &(x, y) in mines {
        game.grid[y][x].content = CellContent::Mine;
    }
    for y in 0..height {
        for x in 0..width {
            if matches!(game.grid[y][x].content, CellContent::Mine) {
                continue;
            }
            let count = neighbours(x, y, width, height)
                .filter(|&(nx, ny)| matches!(game.grid[ny][nx].content, CellContent::Mine))
                .count() as u8;
            game.grid[y][x].content = CellContent::Empty(count);
        }
    }
    game.mines_generated = true;
    game
}

fn neighbours(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let (w, h) = (width as isize, height as isize);
    (-1isize..=1)
        .flat_map(|dy| (-1isize..=1).map(move |dx| (dx, dy)))
        .filter(|&(dx, dy)| dx != 0 || dy != 0)
        .filter_map(move |(dx, dy)| {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            (nx >= 0 && nx < w && ny >= 0 && ny < h).then_some((nx as usize, ny as usize))
        })
}

/// Reveal cells without triggering the cascade, so a test can describe an exact
/// position. Revealing a mine here would be a bug in the test itself.
fn show(game: &mut Minesweeper, cells: &[(usize, usize)]) {
    for &(x, y) in cells {
        assert!(
            !matches!(game.grid[y][x].content, CellContent::Mine),
            "test revealed a mine at ({x}, {y})"
        );
        game.grid[y][x].state = CellState::Visible;
    }
}

/// Exact per-cell mine probabilities by exhaustive enumeration.
///
/// Walks every way of placing the remaining mines among the unopened cells,
/// keeps the layouts that match every visible number, and returns the fraction
/// of those layouts in which each cell holds a mine. Exponential, hence small
/// boards only — that is the point: it is obviously correct.
fn oracle(game: &Minesweeper) -> Vec<Vec<f64>> {
    let hidden: Vec<(usize, usize)> = (0..game.height)
        .flat_map(|y| (0..game.width).map(move |x| (x, y)))
        .filter(|&(x, y)| game.grid[y][x].state != CellState::Visible)
        .collect();

    let visible_mines = (0..game.height)
        .flat_map(|y| (0..game.width).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            game.grid[y][x].state == CellState::Visible
                && matches!(game.grid[y][x].content, CellContent::Mine)
        })
        .count();
    let to_place = game.mines_count - visible_mines;

    // Constraints: each visible number, with the hidden neighbours it covers.
    let constraints: Vec<(Vec<usize>, usize)> = (0..game.height)
        .flat_map(|y| (0..game.width).map(move |x| (x, y)))
        .filter_map(|(x, y)| {
            if game.grid[y][x].state != CellState::Visible {
                return None;
            }
            let CellContent::Empty(required) = game.grid[y][x].content else {
                return None;
            };
            let covered: Vec<usize> = neighbours(x, y, game.width, game.height)
                .filter_map(|(nx, ny)| hidden.iter().position(|&c| c == (nx, ny)))
                .collect();
            Some((covered, required as usize))
        })
        .collect();

    let mut counts = vec![0u64; hidden.len()];
    let mut layouts = 0u64;
    let mut current = vec![false; hidden.len()];

    fn walk(
        start: usize,
        left: usize,
        current: &mut Vec<bool>,
        constraints: &[(Vec<usize>, usize)],
        counts: &mut [u64],
        layouts: &mut u64,
    ) {
        if left == 0 {
            let ok = constraints.iter().all(|(cells, required)| {
                cells.iter().filter(|&&i| current[i]).count() == *required
            });
            if ok {
                *layouts += 1;
                for (i, &mine) in current.iter().enumerate() {
                    if mine {
                        counts[i] += 1;
                    }
                }
            }
            return;
        }
        // Not enough cells left to place the remaining mines.
        if current.len() - start < left {
            return;
        }
        for i in start..current.len() {
            current[i] = true;
            walk(i + 1, left - 1, current, constraints, counts, layouts);
            current[i] = false;
        }
    }

    walk(0, to_place, &mut current, &constraints, &mut counts, &mut layouts);
    assert!(layouts > 0, "oracle found no consistent layout — bad test position");

    let mut probs = vec![vec![0.0; game.width]; game.height];
    for (i, &(x, y)) in hidden.iter().enumerate() {
        probs[y][x] = counts[i] as f64 / layouts as f64;
    }
    probs
}

fn assert_matches_oracle(game: &Minesweeper, label: &str) {
    let expected = oracle(game);
    let actual = ConstraintSearch::new().calculate(game);

    for y in 0..game.height {
        for x in 0..game.width {
            let (e, a) = (expected[y][x], actual[y][x]);
            assert!(a.is_finite(), "{label}: cell ({x}, {y}) is not finite: {a}");
            assert!(
                (e - a).abs() < 1e-9,
                "{label}: cell ({x}, {y}) expected {e:.9}, got {a:.9}"
            );
        }
    }
}

/// A position where the border splits into two regions that share no cell.
/// Decomposition must treat them as independent yet still respect the single
/// global mine budget that couples them.
#[test]
fn matches_oracle_on_split_border() {
    let mut game = board(8, 3, &[(1, 0), (2, 2), (6, 0), (6, 2)]);
    show(&mut game, &[(0, 1), (1, 1), (2, 1), (5, 1), (6, 1), (7, 1)]);
    assert_matches_oracle(&game, "split border");
}

/// One connected border: the case where decomposition cannot help and the DFS
/// has to enumerate the whole thing.
#[test]
fn matches_oracle_on_connected_border() {
    let mut game = board(6, 4, &[(0, 0), (2, 1), (3, 3), (5, 0), (4, 2)]);
    show(&mut game, &[(1, 1), (2, 2), (3, 1), (4, 1), (1, 2), (3, 2)]);
    assert_matches_oracle(&game, "connected border");
}

/// Unconstrained interior cells are handled analytically rather than by the
/// search, so they need their own check against the oracle.
#[test]
fn matches_oracle_with_interior_cells() {
    let mut game = board(6, 5, &[(0, 0), (1, 2), (4, 4), (5, 1)]);
    show(&mut game, &[(0, 1), (1, 1), (2, 1), (2, 2)]);
    assert_matches_oracle(&game, "interior cells");
}

/// Flagged cells are still unknowns to the estimator — a flag is a player's
/// opinion, not evidence.
#[test]
fn flags_do_not_change_probabilities() {
    let mut game = board(6, 4, &[(0, 0), (2, 1), (3, 3), (5, 0), (4, 2)]);
    show(&mut game, &[(1, 1), (2, 2), (3, 1), (4, 1), (1, 2), (3, 2)]);
    let before = ConstraintSearch::new().calculate(&game);

    game.toggle_flag(0, 0);
    game.toggle_flag(5, 3);
    let after = ConstraintSearch::new().calculate(&game);

    assert_eq!(before, after, "flagging a cell changed the estimate");
}

/// Across any position, the estimated mines must add up to the mines that are
/// actually left. This one holds on boards far too large for the oracle.
#[test]
fn probabilities_sum_to_remaining_mines() {
    let mut game = board(
        12,
        12,
        &[
            (0, 0), (3, 1), (5, 4), (7, 2), (9, 9), (11, 0), (2, 7), (6, 6),
            (8, 11), (10, 5), (1, 10), (4, 8),
        ],
    );
    show(
        &mut game,
        &[
            (5, 5), (6, 5), (7, 5), (5, 6), (7, 6), (5, 7), (6, 7), (7, 7),
            (1, 1), (2, 2), (9, 1), (10, 2),
        ],
    );

    let probs = ConstraintSearch::new().calculate(&game);
    let total: f64 = (0..game.height)
        .flat_map(|y| (0..game.width).map(move |x| (x, y)))
        .filter(|&(x, y)| game.grid[y][x].state != CellState::Visible)
        .map(|(x, y)| probs[y][x])
        .sum();

    assert!(
        (total - game.mines_count as f64).abs() < 1e-6,
        "probabilities sum to {total}, expected {}",
        game.mines_count
    );
}

/// A cell that is a mine in every consistent layout must read exactly 1.0, and
/// one that is safe in every layout exactly 0.0 — the auto-reveal feature in
/// every front-end depends on that second guarantee being exact, not merely close.
#[test]
fn certain_cells_are_exact() {
    // The 1 at (1,1) has a single hidden neighbour left, so it must be the mine.
    let mut game = board(4, 3, &[(0, 0)]);
    show(
        &mut game,
        &[(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)],
    );

    let probs = ConstraintSearch::new().calculate(&game);
    assert_eq!(probs[0][0], 1.0, "a certain mine must read exactly 1.0");
    assert_eq!(probs[1][0], 0.0, "a certain safe cell must read exactly 0.0");
    assert_eq!(probs[2][0], 0.0, "a certain safe cell must read exactly 0.0");
}
