/// Training data generator for the neural mine probability predictor.
///
/// Produces JSONL records to stdout, one per game position:
///   { width, height, mines_count, grid: [[{state, content}]], probs: [[f64]] }
///
/// `state`: 0=Hidden, 1=Visible, 2=Flagged
/// `content`: 0..8 for Empty(n) when Visible, 9 for Mine (visible), 255 otherwise
///
/// Run via `python neural/datagen.py N_SAMPLES N_PROCESSES` which
/// spawns this binary in parallel and merges the output streams.
///
/// Every label is exact. Positions the search cannot finish within its budget are
/// skipped rather than labelled with a sampled estimate — a sampled label looks
/// exactly like an exact one in the output, so letting one through would quietly
/// teach the model someone else's guesses.

use std::io::{self, Write};

use rand::Rng;
use serde::Serialize;

use minesweeper_core::probability::{ConstraintSearch, SimUpdate};
use minesweeper_core::{CellContent, CellState, GameState, Minesweeper};

// Board configurations: (width, height, mines)
const CONFIGS: [(usize, usize, usize); 5] = [
    (9, 9, 10),   // beginner
    (10, 10, 15), // custom small
    (16, 16, 40), // intermediate
    (16, 16, 51), // custom intermediate
    (30, 16, 99), // expert
];

#[derive(Serialize)]
struct CellRecord {
    /// 0=Hidden, 1=Visible, 2=Flagged
    state: u8,
    /// 0–8 for visible Empty(n), 9 for visible Mine, 255 for hidden/flagged
    content: u8,
}

#[derive(Serialize)]
struct GameRecord {
    width: usize,
    height: usize,
    mines_count: usize,
    grid: Vec<Vec<CellRecord>>,
    probs: Vec<Vec<f64>>,
}

fn count_hidden(game: &Minesweeper) -> usize {
    (0..game.height)
        .flat_map(|y| (0..game.width).map(move |x| (x, y)))
        .filter(|&(x, y)| game.grid[y][x].state == CellState::Hidden)
        .count()
}

/// Exact per-cell probabilities, or `None` when the search could not finish.
fn exact_labels(cs: &ConstraintSearch, game: &Minesweeper) -> Option<Vec<Vec<f64>>> {
    let (tx, rx) = std::sync::mpsc::channel();
    cs.calculate_with_progress(game, tx);
    while let Ok(update) = rx.recv() {
        if let SimUpdate::Done { valid, probs, .. } = update {
            // Zero layouts is how the search reports that it has no trustworthy
            // answer, whether from an exhausted budget or an unusable one.
            return (valid > 0).then_some(probs);
        }
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1000);

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    let mut rng = rand::thread_rng();
    // Budgeted rather than exhaustive, and kept across positions so its cache of
    // region solutions carries over. Unbounded would risk one pathological
    // position stalling a worker for minutes in an unattended run.
    let cs = ConstraintSearch::new();
    let mut generated = 0;

    while generated < n {
        let &(w, h, mines) = &CONFIGS[rng.gen_range(0..CONFIGS.len())];
        let mut game = Minesweeper::new(w, h, mines);

        // First click — reveal a random cell to place mines safely.
        let first_x = rng.gen_range(0..w);
        let first_y = rng.gen_range(0..h);
        game.reveal(first_x, first_y);

        if game.state != GameState::Playing {
            continue;
        }

        // Simulate a partial game with random moves only (fast).
        // We want diversity in board state, not play quality.
        //
        // Every stage of a game is wanted, not just the end: this used to stop as
        // soon as few enough cells were left for the labeller to keep up, which
        // meant the model only ever saw endgames.
        let max_moves = rng.gen_range(1..=(w * h / 4).max(2));
        for _ in 0..max_moves {
            if game.state != GameState::Playing {
                break;
            }

            let hidden: Vec<(usize, usize)> = (0..h)
                .flat_map(|y| (0..w).map(move |x| (x, y)))
                .filter(|&(x, y)| game.grid[y][x].state == CellState::Hidden)
                .collect();
            if hidden.is_empty() {
                break;
            }
            let (cx, cy) = hidden[rng.gen_range(0..hidden.len())];
            game.reveal(cx, cy);
        }

        // Skip finished games.
        if game.state != GameState::Playing {
            continue;
        }

        if count_hidden(&game) == 0 {
            continue;
        }

        // Exact probability labels, or nothing. `calculate` would fall back to
        // sampling when the search runs out of budget; going through the channel
        // lets us see that it did and drop the position instead.
        let Some(probs) = exact_labels(&cs, &game) else {
            continue;
        };

        // Build the serializable grid.
        let grid: Vec<Vec<CellRecord>> = (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| {
                        let cell = &game.grid[y][x];
                        let state = match cell.state {
                            CellState::Hidden => 0,
                            CellState::Visible => 1,
                            CellState::Flagged => 2,
                        };
                        let content = if cell.state == CellState::Visible {
                            match cell.content {
                                CellContent::Empty(n) => n,
                                CellContent::Mine => 9,
                            }
                        } else {
                            255
                        };
                        CellRecord { state, content }
                    })
                    .collect()
            })
            .collect();

        let record = GameRecord {
            width: w,
            height: h,
            mines_count: mines,
            grid,
            probs,
        };

        if let Ok(line) = serde_json::to_string(&record) {
            let _ = writeln!(out, "{}", line);
            generated += 1;
        }
    }
}
