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

use crate::Minesweeper;

use super::setup::{build_probs, memory_estimate, SimSetup};
use super::patch::{PatchSource, N_CHANNELS, PATCH, PATCH_LEN};
use super::{SimUpdate, Strategy};


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

    /// Load and optimise the ONNX model from bytes already in memory.
    ///
    /// For callers with no filesystem to read from — the WebAssembly build fetches
    /// the model over HTTP and hands it straight over.
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(bytes))
            .context("parsing the ONNX model")?
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

        let memory_bytes = memory_estimate(&setup);
        let hidden = &setup.hidden_cells;
        let n_hidden = hidden.len();

        if n_hidden == 0 {
            send_done(build_probs(&[], 0.0, &setup, game.width, game.height), 0, memory_bytes);
            return;
        }

        // One description of the board, shared with the hand-written network so
        // the patch layout has a single definition on the Rust side.
        let source = PatchSource::new(game);

        // Flat batch: (n_hidden, N_CHANNELS, PATCH, PATCH) in C order.
        let mut batch = vec![0.0f32; n_hidden * PATCH_LEN];
        for (i, &(cx, cy)) in hidden.iter().enumerate() {
            source.fill(cx, cy, &mut batch[i * PATCH_LEN..][..PATCH_LEN]);
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

