/// Neural network mine probability estimator (ONNX via `tract-onnx`).
///
/// Loads a pre-trained PatchCNN model from an ONNX file and runs a single
/// batched forward pass for all hidden cells in the current board state.
///
/// # Input representation
///
/// For each hidden cell at `(cx, cy)` we extract a 9×9 patch (HALF=4) with
/// 8 channels:
///
/// | Ch | Value                          | Description                        |
/// |----|--------------------------------|------------------------------------|
/// | 0  | mine_count / 8.0 if Visible    | Numbered cell value                |
/// | 1  | {0,1}                          | Is patch cell Visible              |
/// | 2  | {0,1}                          | Is patch cell Hidden               |
/// | 3  | {0,1}                          | Is patch cell Flagged              |
/// | 4  | {0,1}                          | Is patch cell out-of-bounds        |
/// | 5  | 1.0 at (4,4) else 0            | Target cell marker                 |
/// | 6  | mines_remaining / total_hidden | Global ratio broadcast             |
/// | 7  | {0,1}                          | Is patch cell a border hidden cell |
///
/// # Priority
///
/// `Strategy::NeuralNetwork` has priority 1 (same as Monte Carlo), so
/// ConstraintSearch (priority 2) will override it once done.
use std::sync::mpsc::Sender;

use anyhow::Context;
use tract_onnx::prelude::*;

use crate::{CellContent, CellState, Minesweeper};

use super::monte_carlo::{build_probs, mc_memory_estimate, SimSetup};
use super::{SimUpdate, Strategy};

const HALF: usize = 4;
const PATCH: usize = 2 * HALF + 1; // 9
const N_CHANNELS: usize = 8;

type OnnxModel = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Neural network strategy backed by a tract-onnx model.
pub struct NeuralNetwork {
    model: OnnxModel,
}

impl NeuralNetwork {
    /// Load and optimise the ONNX model from `model_path`.
    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .with_context(|| format!("loading ONNX model from {model_path}"))?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }

    /// Run inference and send a single `SimUpdate::Done` through `tx`.
    pub fn calculate_with_progress(&self, game: &Minesweeper, tx: Sender<SimUpdate>) {
        let send_done = |probs, valid, memory_bytes| {
            let _ = tx.send(SimUpdate::Done {
                strategy: Strategy::NeuralNetwork,
                attempts: 1,
                valid,
                memory_bytes,
                probs,
            });
        };

        let Some(setup) = SimSetup::build(game) else {
            send_done(vec![vec![0.0; game.width]; game.height], 0, 0);
            return;
        };

        let memory_bytes = mc_memory_estimate(&setup);
        let hidden = &setup.hidden_cells;
        let n_hidden = hidden.len();

        if n_hidden == 0 {
            send_done(build_probs(&[], 0.0, &setup, game.width, game.height), 0, memory_bytes);
            return;
        }

        // Pre-compute state/content arrays and border mask.
        let grid_state = build_grid_state(game);
        let grid_content = build_grid_content(game);
        let border_mask = compute_border_mask(game, &grid_state, &grid_content);

        let visible_mines: usize = (0..game.height)
            .flat_map(|y| (0..game.width).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                matches!(game.grid[y][x].content, CellContent::Mine)
                    && game.grid[y][x].state == CellState::Visible
            })
            .count();
        let mines_ratio = game.mines_count.saturating_sub(visible_mines) as f32
            / n_hidden.max(1) as f32;

        // Build flat batch buffer: shape (n_hidden, N_CHANNELS, PATCH, PATCH) in C order.
        let n_elem = n_hidden * N_CHANNELS * PATCH * PATCH;
        let mut batch = vec![0.0f32; n_elem];
        for (i, &(cx, cy)) in hidden.iter().enumerate() {
            let base = i * N_CHANNELS * PATCH * PATCH;
            extract_patch(
                &grid_state,
                &grid_content,
                &border_mask,
                cx,
                cy,
                mines_ratio,
                game.width,
                game.height,
                &mut batch[base..base + N_CHANNELS * PATCH * PATCH],
            );
        }

        // Wrap in a tract Tensor.
        let shape = [n_hidden, N_CHANNELS, PATCH, PATCH];
        let input = match tract_ndarray::Array::from_shape_vec(shape, batch) {
            Ok(arr) => arr.into_tensor(),
            Err(e) => {
                eprintln!("NeuralNetwork: failed to build input tensor: {e}");
                return;
            }
        };

        let outputs = match self.model.run(tvec![input.into()]) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("NeuralNetwork: inference error: {e}");
                return;
            }
        };

        // Extract flat probability vector.
        let probs_flat: Vec<f32> = match outputs[0].to_array_view::<f32>() {
            Ok(view) => view.iter().copied().collect(),
            Err(e) => {
                eprintln!("NeuralNetwork: failed to extract output: {e}");
                return;
            }
        };

        // Map flat probs back to 2D grid.  We pass total_weight=1.0 so
        // build_probs treats mine_counts[i] directly as P(mine).
        let mine_counts: Vec<f64> = probs_flat
            .iter()
            .take(n_hidden)
            .map(|&p| p.clamp(0.0, 1.0) as f64)
            .collect();

        let probs = build_probs(&mine_counts, 1.0, &setup, game.width, game.height);
        send_done(probs, n_hidden, memory_bytes);
    }
}

// ---------------------------------------------------------------------------
// Patch extraction
// ---------------------------------------------------------------------------

fn build_grid_state(game: &Minesweeper) -> Vec<Vec<u8>> {
    (0..game.height)
        .map(|y| {
            (0..game.width)
                .map(|x| match game.grid[y][x].state {
                    CellState::Hidden => 0,
                    CellState::Visible => 1,
                    CellState::Flagged => 2,
                })
                .collect()
        })
        .collect()
}

fn build_grid_content(game: &Minesweeper) -> Vec<Vec<u8>> {
    (0..game.height)
        .map(|y| {
            (0..game.width)
                .map(|x| {
                    if game.grid[y][x].state == CellState::Visible {
                        match game.grid[y][x].content {
                            CellContent::Empty(n) => n,
                            CellContent::Mine => 9,
                        }
                    } else {
                        255
                    }
                })
                .collect()
        })
        .collect()
}

/// Fill `out[0..N_CHANNELS*PATCH*PATCH]` with the 8-channel patch for cell `(cx,cy)`.
/// Layout: channel-major (C×H×W), i.e. `out[ch * PATCH*PATCH + pi*PATCH + pj]`.
fn extract_patch(
    grid_state: &[Vec<u8>],
    grid_content: &[Vec<u8>],
    border_mask: &[Vec<bool>],
    cx: usize,
    cy: usize,
    mines_ratio: f32,
    width: usize,
    height: usize,
    out: &mut [f32],
) {
    let stride = PATCH * PATCH;
    for pi in 0..PATCH {
        for pj in 0..PATCH {
            let gy = cy as isize + pi as isize - HALF as isize;
            let gx = cx as isize + pj as isize - HALF as isize;
            let pos = pi * PATCH + pj;

            if gy < 0 || gy >= height as isize || gx < 0 || gx >= width as isize {
                out[4 * stride + pos] = 1.0; // out-of-bounds
                continue;
            }
            let gy = gy as usize;
            let gx = gx as usize;
            match grid_state[gy][gx] {
                1 => {
                    out[1 * stride + pos] = 1.0;
                    let c = grid_content[gy][gx];
                    if c < 9 {
                        out[0 * stride + pos] = c as f32 / 8.0;
                    }
                }
                0 => out[2 * stride + pos] = 1.0,
                2 => out[3 * stride + pos] = 1.0,
                _ => {}
            }
            if border_mask[gy][gx] {
                out[7 * stride + pos] = 1.0;
            }
        }
    }
    out[5 * stride + HALF * PATCH + HALF] = 1.0; // centre marker
    for pos in 0..stride {
        out[6 * stride + pos] = mines_ratio; // global ratio broadcast
    }
}

/// `true` for hidden/flagged cells adjacent to a visible numbered cell.
fn compute_border_mask(
    game: &Minesweeper,
    grid_state: &[Vec<u8>],
    grid_content: &[Vec<u8>],
) -> Vec<Vec<bool>> {
    let mut mask = vec![vec![false; game.width]; game.height];
    for y in 0..game.height {
        for x in 0..game.width {
            let s = grid_state[y][x];
            if s != 0 && s != 2 {
                continue;
            }
            'outer: for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let ny = y as isize + dy;
                    let nx = x as isize + dx;
                    if ny < 0
                        || ny >= game.height as isize
                        || nx < 0
                        || nx >= game.width as isize
                    {
                        continue;
                    }
                    let ny = ny as usize;
                    let nx = nx as usize;
                    if grid_state[ny][nx] == 1 {
                        let c = grid_content[ny][nx];
                        if c > 0 && c < 9 {
                            mask[y][x] = true;
                            break 'outer;
                        }
                    }
                }
            }
        }
    }
    mask
}
