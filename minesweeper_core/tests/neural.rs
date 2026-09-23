//! End-to-end check of the ONNX estimator.
//!
//! The Rust side of the neural feature is the last link in a long chain — Rust
//! datagen, Python preparation, training, ONNX export, `tract` — and every step
//! before this one can succeed while this one silently does not. Notably the
//! patch layout is written down twice, in `neural.rs` and in `neural/dataset.py`,
//! with nothing tying them together: a model trained against one and served by
//! the other returns confident nonsense rather than an error.
//!
//! Requires a trained model, which is not in the repository, so it is skipped
//! unless one is pointed at:
//!
//! ```bash
//! MINESWEEPER_ONNX=neural/onnx/model.onnx cargo test -p minesweeper_core --features neural
//! ```

#![cfg(feature = "neural")]

use std::sync::mpsc::channel;

use minesweeper_core::probability::{NeuralNetwork, SimUpdate};
use minesweeper_core::{CellContent, CellState, Minesweeper};

fn model_path() -> Option<String> {
    match std::env::var("MINESWEEPER_ONNX") {
        Ok(path) if std::path::Path::new(&path).exists() => Some(path),
        _ => {
            eprintln!("skipping: set MINESWEEPER_ONNX to a trained model to run this");
            None
        }
    }
}

/// A mid-game board, built explicitly so the test does not depend on an RNG.
fn position() -> Minesweeper {
    let mines = [(0, 0), (2, 1), (3, 3), (5, 0), (4, 2)];
    let (width, height) = (6, 4);
    let mut game = Minesweeper::new(width, height, mines.len());
    for &(x, y) in &mines {
        game.grid[y][x].content = CellContent::Mine;
    }
    for y in 0..height {
        for x in 0..width {
            if matches!(game.grid[y][x].content, CellContent::Mine) {
                continue;
            }
            let mut count = 0u8;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if (0..width as i32).contains(&nx) && (0..height as i32).contains(&ny) {
                        if matches!(game.grid[ny as usize][nx as usize].content, CellContent::Mine) {
                            count += 1;
                        }
                    }
                }
            }
            game.grid[y][x].content = CellContent::Empty(count);
        }
    }
    game.mines_generated = true;
    for &(x, y) in &[(1, 1), (2, 2), (3, 1), (4, 1), (1, 2), (3, 2)] {
        game.grid[y][x].state = CellState::Visible;
    }
    game
}

#[test]
fn the_model_loads_and_predicts() {
    let Some(path) = model_path() else { return };

    let network = NeuralNetwork::new(&path).expect("loading the ONNX model");
    let game = position();

    let (tx, rx) = channel();
    network.calculate_with_progress(&game, tx);

    let mut probs = None;
    while let Ok(update) = rx.recv() {
        if let SimUpdate::Done { probs: p, .. } = update {
            probs = Some(p);
            break;
        }
    }
    let probs = probs.expect("the estimator reported no result");

    assert_eq!(probs.len(), game.height);
    for y in 0..game.height {
        assert_eq!(probs[y].len(), game.width);
        for x in 0..game.width {
            let p = probs[y][x];
            assert!(p.is_finite(), "({x}, {y}) is not finite: {p}");
            assert!((0.0..=1.0).contains(&p), "({x}, {y}) is outside [0, 1]: {p}");
            if game.grid[y][x].state == CellState::Visible {
                assert_eq!(p, 0.0, "({x}, {y}) is already open but was given {p}");
            }
        }
    }

    // A model that predicts is not the same as a model that is right, but one
    // that has learned nothing outputs the same number everywhere. This is the
    // cheapest check that the patch layout the Rust side builds is the one the
    // model was trained on: disagree about a channel and the input carries no
    // signal, so the output goes flat.
    let hidden: Vec<f64> = (0..game.height)
        .flat_map(|y| (0..game.width).map(move |x| (x, y)))
        .filter(|&(x, y)| game.grid[y][x].state == CellState::Hidden)
        .map(|(x, y)| probs[y][x])
        .collect();
    let spread = hidden.iter().cloned().fold(f64::MIN, f64::max)
        - hidden.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        spread > 0.01,
        "every unopened cell got essentially the same probability (spread {spread:.4}) — \
         the model is untrained, or the patch layout here does not match the one it saw"
    );
}
