//! The PatchCNN, written out by hand.
//!
//! Same network as the ONNX estimator, reading the same weights, with no
//! dependency behind it. There are two reasons for that.
//!
//! **Size.** Running the ONNX file needs `tract`, and compiling `tract` into the
//! WebAssembly build takes the module from 174 KB to 13.4 MB — seventy-seven times
//! larger, on a page whose whole point is loading instantly from a CDN. The
//! architecture is four convolutions and three linear layers; spelling it out
//! costs a few hundred lines and no bytes at all.
//!
//! **Learning during play.** Owning the arithmetic means owning the gradient. The
//! exact solver labels every position it solves, for free and correctly, so the
//! network can be corrected as the game goes on rather than only between training
//! runs — see [`PatchCnn::learn`].
//!
//! Weights come from `neural/export_weights.py`, which folds each BatchNorm into
//! the convolution ahead of it, so inference here is convolution, bias, ReLU and
//! nothing else. `neural/patchcnn_reference.py` is the same thing in numpy and is
//! what this is tested against.

use super::patch::{PatchSource, N_CHANNELS, PATCH, PATCH_LEN};
use crate::{CellState, Minesweeper};

/// Identifies the weight layout. Any change to the architecture changes this.
const MAGIC: &[u8; 8] = b"MSPCNN1\0";

/// (outputs, inputs) per convolution, all 3x3 with one cell of padding.
const CONVS: [(usize, usize); 4] = [(32, 8), (64, 32), (64, 64), (32, 64)];
/// (outputs, inputs) per linear layer.
const LINEARS: [(usize, usize); 3] = [(128, 289), (64, 128), (1, 64)];
/// The convolution stack pools to 3x3 before the linear layers.
const POOL: usize = 3;

/// What went wrong reading a weights file.
#[derive(Debug, PartialEq, Eq)]
pub enum WeightsError {
    /// Not a PatchCNN weights file, or a layout this build does not know.
    BadMagic,
    /// The file holds a different number of values than the architecture wants.
    WrongLength { found: usize, expected: usize },
}

impl std::fmt::Display for WeightsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not a PatchCNN weights file"),
            Self::WrongLength { found, expected } => write!(
                f,
                "weights file holds {found} values, this network wants {expected}"
            ),
        }
    }
}

impl std::error::Error for WeightsError {}

struct Layer {
    outputs: usize,
    inputs: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

/// A trained PatchCNN.
///
/// `Debug` prints the shape rather than 122k weights, which no one wants to read.
pub struct PatchCnn {
    convs: Vec<Layer>,
    linears: Vec<Layer>,
}

impl std::fmt::Debug for PatchCnn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let values: usize = self
            .convs
            .iter()
            .chain(&self.linears)
            .map(|l| l.weight.len() + l.bias.len())
            .sum();
        write!(
            f,
            "PatchCnn {{ {} convolutions, {} linear layers, {values} weights }}",
            self.convs.len(),
            self.linears.len()
        )
    }
}

impl PatchCnn {
    /// Read the weights written by `neural/export_weights.py`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WeightsError> {
        if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
            return Err(WeightsError::BadMagic);
        }
        let values: Vec<f32> = bytes[MAGIC.len()..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let expected: usize = CONVS
            .iter()
            .map(|(o, i)| o * i * 9 + o)
            .chain(LINEARS.iter().map(|(o, i)| o * i + o))
            .sum();
        if values.len() != expected {
            return Err(WeightsError::WrongLength { found: values.len(), expected });
        }

        let mut at = 0;
        let mut take = |count: usize| {
            let slice = values[at..at + count].to_vec();
            at += count;
            slice
        };
        let convs = CONVS
            .iter()
            .map(|&(outputs, inputs)| Layer {
                outputs,
                inputs,
                weight: take(outputs * inputs * 9),
                bias: take(outputs),
            })
            .collect();
        let linears = LINEARS
            .iter()
            .map(|&(outputs, inputs)| Layer {
                outputs,
                inputs,
                weight: take(outputs * inputs),
                bias: take(outputs),
            })
            .collect();

        Ok(Self { convs, linears })
    }

    /// P(mine) for one patch, which must hold [`PATCH_LEN`] values.
    pub fn predict(&self, patch: &[f32]) -> f32 {
        self.run(patch).probability
    }

    /// Forward pass, keeping what a later gradient step needs.
    fn run(&self, patch: &[f32]) -> Activations {
        debug_assert_eq!(patch.len(), PATCH_LEN);

        let mut current = patch.to_vec();
        let mut channels = N_CHANNELS;
        for layer in &self.convs {
            current = relu(convolve(&current, channels, PATCH, layer));
            channels = layer.outputs;
        }

        // Adaptive average pool to 3x3: 9 divides by 3, so every output is the
        // mean of one 3x3 block.
        let block = PATCH / POOL;
        let mut features = Vec::with_capacity(channels * POOL * POOL + 1);
        for channel in 0..channels {
            for by in 0..POOL {
                for bx in 0..POOL {
                    let mut sum = 0.0;
                    for y in 0..block {
                        for x in 0..block {
                            sum += current
                                [channel * PATCH * PATCH + (by * block + y) * PATCH + bx * block + x];
                        }
                    }
                    features.push(sum / (block * block) as f32);
                }
            }
        }
        // The global mines ratio, which channel 6 carries uniformly.
        features.push(patch[6 * PATCH * PATCH + 4 * PATCH + 4]);

        let head_input = features.clone();
        let hidden1 = relu(linear(&features, &self.linears[0]));
        let hidden2 = relu(linear(&hidden1, &self.linears[1]));
        let logit = linear(&hidden2, &self.linears[2])[0];

        Activations {
            head_input,
            hidden2,
            probability: 1.0 / (1.0 + (-logit).exp()),
        }
    }

    /// Nudge the output layer towards `target` for this patch, and return the
    /// error before the step.
    ///
    /// Only the last layer moves. The convolutions stay as they were trained,
    /// because updating them from single positions arriving in game order — highly
    /// correlated, one board at a time — is how a network forgets what it knew.
    /// The final layer is a logistic regression over fixed features, which is
    /// stable under exactly those conditions.
    ///
    /// The gradient is the simple one: for a sigmoid output under log loss,
    /// d(loss)/d(logit) is just `prediction - target`.
    pub fn learn(&mut self, patch: &[f32], target: f32, rate: f32) -> f32 {
        let activations = self.run(patch);
        let error = activations.probability - target;

        let output = self.linears.last_mut().expect("the head is always present");
        for (weight, feature) in output.weight.iter_mut().zip(&activations.hidden2) {
            *weight -= rate * error * feature;
        }
        output.bias[0] -= rate * error;

        let _ = activations.head_input; // kept for a future deeper update
        error.abs()
    }

    /// P(mine) for every unopened cell, as a grid. Opened cells read 0.
    pub fn calculate(&self, game: &Minesweeper) -> Vec<Vec<f64>> {
        let source = PatchSource::new(game);
        let mut probs = vec![vec![0.0f64; game.width]; game.height];
        let mut patch = vec![0.0f32; PATCH_LEN];

        for y in 0..game.height {
            for x in 0..game.width {
                if game.grid[y][x].state == CellState::Visible {
                    continue;
                }
                patch.iter_mut().for_each(|v| *v = 0.0);
                source.fill(x, y, &mut patch);
                probs[y][x] = self.predict(&patch) as f64;
            }
        }
        probs
    }

    /// Learn from a board the exact solver has already scored.
    ///
    /// This is the whole point of keeping the arithmetic: a solved position is a
    /// perfectly labelled training example that cost nothing extra to produce.
    /// Returns the mean error over the cells it learned from, before the step.
    pub fn learn_from_board(
        &mut self,
        game: &Minesweeper,
        exact: &[Vec<f64>],
        rate: f32,
    ) -> Option<f32> {
        let source = PatchSource::new(game);
        let mut patch = vec![0.0f32; PATCH_LEN];
        let mut total = 0.0;
        let mut seen = 0usize;

        for y in 0..game.height {
            for x in 0..game.width {
                if game.grid[y][x].state != CellState::Hidden {
                    continue;
                }
                patch.iter_mut().for_each(|v| *v = 0.0);
                source.fill(x, y, &mut patch);
                total += self.learn(&patch, exact[y][x] as f32, rate);
                seen += 1;
            }
        }

        (seen > 0).then(|| total / seen as f32)
    }
}

/// What the forward pass produced, and the intermediates a step back needs.
struct Activations {
    head_input: Vec<f32>,
    hidden2: Vec<f32>,
    probability: f32,
}

/// 3x3 convolution, stride 1, one cell of zero padding.
///
/// Written for the compiler rather than for the reader: the input is copied into
/// a padded buffer so the inner loops carry no bounds tests, and the innermost
/// one walks a contiguous row so it can be vectorised. The obvious version —
/// bounds-checking every one of the nine taps — was eight times slower, which on
/// a 50x50 board is the difference between two seconds and nineteen.
fn convolve(input: &[f32], in_channels: usize, size: usize, layer: &Layer) -> Vec<f32> {
    debug_assert_eq!(in_channels, layer.inputs);
    let plane = size * size;
    let padded_size = size + 2;
    let padded_plane = padded_size * padded_size;

    let mut padded = vec![0.0f32; in_channels * padded_plane];
    for ic in 0..in_channels {
        for y in 0..size {
            let from = &input[ic * plane + y * size..][..size];
            let to = &mut padded[ic * padded_plane + (y + 1) * padded_size + 1..][..size];
            to.copy_from_slice(from);
        }
    }

    let mut out = vec![0.0f32; layer.outputs * plane];
    for oc in 0..layer.outputs {
        let row_out = &mut out[oc * plane..][..plane];
        row_out.fill(layer.bias[oc]);

        for ic in 0..in_channels {
            let kernel = &layer.weight[(oc * in_channels + ic) * 9..][..9];
            let source = &padded[ic * padded_plane..][..padded_plane];

            for ky in 0..3 {
                for kx in 0..3 {
                    let tap = kernel[ky * 3 + kx];
                    if tap == 0.0 {
                        continue;
                    }
                    for y in 0..size {
                        let from = &source[(y + ky) * padded_size + kx..][..size];
                        let to = &mut row_out[y * size..][..size];
                        for (o, v) in to.iter_mut().zip(from) {
                            *o += tap * v;
                        }
                    }
                }
            }
        }
    }
    out
}

fn linear(input: &[f32], layer: &Layer) -> Vec<f32> {
    debug_assert_eq!(input.len(), layer.inputs);
    (0..layer.outputs)
        .map(|o| {
            let row = &layer.weight[o * layer.inputs..][..layer.inputs];
            layer.bias[o] + row.iter().zip(input).map(|(w, v)| w * v).sum::<f32>()
        })
        .collect()
}

fn relu(mut values: Vec<f32>) -> Vec<f32> {
    values.iter_mut().for_each(|v| *v = v.max(0.0));
    values
}
