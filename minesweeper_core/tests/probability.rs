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

use minesweeper_core::probability::{certain_cells, ConstraintSearch, ProbabilityStrategy};
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

/// Deterministic PRNG, so the sweeps below are reproducible and need no dependency.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Build a random board and play it forward, revealing everything propagation
/// proves safe and otherwise guessing a cell that happens not to be a mine, so
/// the sweep reaches deep mid-game positions rather than dying on move three.
fn sweep(seed: u64, width: usize, height: usize, mines: usize, mut check: impl FnMut(&Minesweeper)) {
    let mut rng = Lcg(seed);
    let mut placed: Vec<(usize, usize)> = Vec::new();
    while placed.len() < mines {
        let cell = (rng.below(width), rng.below(height));
        if !placed.contains(&cell) {
            placed.push(cell);
        }
    }
    let mut game = board(width, height, &placed);

    // Opening move: a cell that is not a mine.
    let safe_start = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .find(|&(x, y)| !matches!(game.grid[y][x].content, CellContent::Mine))
        .expect("board is entirely mines");
    game.reveal(safe_start.0, safe_start.1);

    for _ in 0..40 {
        check(&game);

        let proven = certain_cells(&game);
        let opened: Vec<(usize, usize)> = proven
            .safe
            .iter()
            .copied()
            .filter(|&(x, y)| game.grid[y][x].state == CellState::Hidden)
            .collect();
        if !opened.is_empty() {
            for (x, y) in opened {
                game.reveal(x, y);
            }
            continue;
        }

        // Propagation is exhausted — guess a non-mine cell to make progress.
        let guess = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .find(|&(x, y)| {
                game.grid[y][x].state == CellState::Hidden
                    && !matches!(game.grid[y][x].content, CellContent::Mine)
            });
        match guess {
            Some((x, y)) => game.reveal(x, y),
            None => break,
        }
    }
}

/// The safety property the auto-reveal feature now rests on: a cell that
/// propagation calls safe is never a mine, and one it calls a mine always is.
/// Checked across many random mid-game positions on boards far too large for the
/// brute-force oracle.
#[test]
fn propagation_is_never_wrong() {
    for seed in 0..40u64 {
        sweep(seed * 7 + 1, 16, 16, 40, |game| {
            let proven = certain_cells(game);
            for &(x, y) in &proven.safe {
                assert!(
                    !matches!(game.grid[y][x].content, CellContent::Mine),
                    "propagation called ({x}, {y}) safe but it is a mine"
                );
            }
            for &(x, y) in &proven.mines {
                assert!(
                    matches!(game.grid[y][x].content, CellContent::Mine),
                    "propagation called ({x}, {y}) a mine but it is not"
                );
            }
        });
    }
}

/// Same property on the board size that was hanging, where the estimators are
/// too slow to consult but propagation still has to be right.
#[test]
fn propagation_is_never_wrong_on_a_dense_board() {
    for seed in 0..12u64 {
        sweep(seed * 13 + 5, 30, 30, 250, |game| {
            for &(x, y) in &certain_cells(game).safe {
                assert!(
                    !matches!(game.grid[y][x].content, CellContent::Mine),
                    "propagation called ({x}, {y}) safe but it is a mine"
                );
            }
        });
    }
}

/// Propagation is allowed to find less than the full search, but never something
/// different: anything it proves must match the exact probabilities.
#[test]
fn propagation_agrees_with_the_exact_search() {
    for seed in 0..15u64 {
        sweep(seed * 11 + 3, 10, 10, 15, |game| {
            let proven = certain_cells(game);
            if proven.safe.is_empty() && proven.mines.is_empty() {
                return;
            }
            let probs = ConstraintSearch::new().calculate(game);
            for &(x, y) in &proven.safe {
                assert_eq!(probs[y][x], 0.0, "({x}, {y}) proven safe but estimated non-zero");
            }
            for &(x, y) in &proven.mines {
                assert_eq!(probs[y][x], 1.0, "({x}, {y}) proven a mine but not estimated 1.0");
            }
        });
    }
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

/// Regression: a sparse 50x50 board detonated auto-reveal in the wasm front-end,
/// which only ever opens cells propagation calls safe.
#[test]
fn propagation_is_never_wrong_on_a_large_sparse_board() {
    for seed in 0..6u64 {
        sweep(seed * 977 + 1, 50, 50, 150, |game| {
            for &(x, y) in &certain_cells(game).safe {
                assert!(
                    !matches!(game.grid[y][x].content, CellContent::Mine),
                    "propagation called ({x}, {y}) safe but it is a mine"
                );
            }
        });
    }
}

/// Regression for a scaling bug that made every cell on a large sparse board
/// read 0%. The leaf weights were divided by the largest binomial in the table
/// rather than the largest one the search can reach, so the weights that were
/// actually used underflowed to zero. A grid of zeros means "all safe", which
/// detonated auto-reveal.
///
/// The mines-sum invariant catches it immediately: however the weights are
/// scaled, the estimates must still add up to the mines that are out there.
#[test]
fn probabilities_stay_calibrated_on_large_boards() {
    for (width, height, mines) in [(50, 50, 150), (40, 40, 60), (30, 30, 99)] {
        sweep(width as u64 * 31 + mines as u64, width, height, mines, |game| {
            let probs = ConstraintSearch::new().calculate(game);
            let total: f64 = (0..game.height)
                .flat_map(|y| (0..game.width).map(move |x| (x, y)))
                .filter(|&(x, y)| game.grid[y][x].state != CellState::Visible)
                .map(|(x, y)| probs[y][x])
                .sum();
            assert!(
                (total - mines as f64).abs() < 0.01,
                "{width}x{height}/{mines}: probabilities sum to {total:.4}, expected {mines}"
            );
        });
    }
}

/// The decomposition and the convolution that reassembles it are subtle enough
/// that a handful of hand-built positions is not convincing. This walks many
/// random mid-game positions on boards small enough to enumerate exhaustively
/// and demands the solver agree with brute force everywhere, to within floating
/// point noise.
///
/// Board sizes are kept small because the oracle is exponential in the number of
/// unopened cells — that is the price of an obviously-correct reference.
#[test]
fn matches_oracle_across_random_positions() {
    let mut checked = 0;
    for seed in 0..25u64 {
        for (width, height, mines) in [(6, 4, 5), (7, 4, 6), (5, 5, 6)] {
            sweep(seed * 131 + width as u64, width, height, mines, |game| {
                let hidden = (0..game.height)
                    .flat_map(|y| (0..game.width).map(move |x| (x, y)))
                    .filter(|&(x, y)| game.grid[y][x].state != CellState::Visible)
                    .count();
                // Keep the oracle's work bounded, and skip positions where
                // nothing is open yet (no constraints to decompose).
                if hidden > 18 || hidden == game.width * game.height {
                    return;
                }
                assert_matches_oracle(game, &format!("{width}x{height}/{mines} seed {seed}"));
                checked += 1;
            });
        }
    }
    assert!(checked > 100, "only {checked} positions checked — sweep is not exercising much");
}

/// Independent regions must be combined through the global mine budget, not
/// treated as if each had its own. A position with two separated regions and few
/// enough mines that they compete for them is where a wrong combination shows up.
#[test]
fn separated_regions_compete_for_the_same_mines() {
    // Two 1-cell-wide corridors far apart, and two mines to share between them.
    let mut game = board(11, 3, &[(1, 1), (9, 1)]);
    show(
        &mut game,
        &[
            (0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2),
            (8, 0), (9, 0), (10, 0), (8, 1), (10, 1), (8, 2), (9, 2), (10, 2),
        ],
    );
    assert_matches_oracle(&game, "two corridors");

    // Each corridor's single unopened cell must be a certain mine: its numbers
    // leave no alternative, and the two mines are exactly accounted for.
    let probs = ConstraintSearch::new().calculate(&game);
    assert_eq!(probs[1][1], 1.0);
    assert_eq!(probs[1][9], 1.0);
}
