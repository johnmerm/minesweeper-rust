use crate::Minesweeper;

pub mod monte_carlo;
pub mod components;
pub(crate) mod patch;
pub mod patch_cnn;
pub mod constraint_search;
#[cfg(feature = "neural")] pub mod neural;
pub use monte_carlo::MonteCarlo;
pub use constraint_search::ConstraintSearch;
pub use patch_cnn::{PatchCnn, WeightsError};
#[cfg(feature = "neural")] pub use neural::NeuralNetwork;

pub trait ProbabilityStrategy {
    fn calculate(&self, game: &Minesweeper) -> Vec<Vec<f64>>;
}

/// Cells whose contents follow from the visible numbers alone.
///
/// Returned by [`certain_cells`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CertainCells {
    /// Unopened cells that must be mines.
    pub mines: Vec<(usize, usize)>,
    /// Unopened cells that cannot be mines, and are therefore safe to reveal.
    pub safe: Vec<(usize, usize)>,
}

/// Cells that constraint propagation alone proves are mines or safe.
///
/// This is the cheap half of what the estimators do: it applies the local rules
/// — "this 3 already touches 3 unopened cells, so they are all mines", "this 1
/// already touches its mine, so its other neighbours are safe" — to a fixpoint,
/// without enumerating or sampling any layout. It is polynomial where the
/// estimators are exponential.
///
/// It is deliberately *incomplete*: a cell can be provably safe while no
/// sequence of local rules shows it, and only a full search finds those. What it
/// returns is always sound, though, so a caller can reveal every cell in `safe`
/// without risk. Use it to make progress cheaply, and fall back to a full
/// strategy when it runs dry.
pub fn certain_cells(game: &Minesweeper) -> CertainCells {
    match monte_carlo::SimSetup::build(game) {
        Some(setup) => CertainCells {
            mines: setup.certain_mines,
            safe: setup.certain_safe,
        },
        None => CertainCells::default(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Strategy {
    MonteCarlo,
    ConstraintSearch,
    /// Neural network estimator (ONNX), priority 1 — same as MC.
    /// ConstraintSearch (priority 2) overrides it once it finishes.
    #[cfg(feature = "neural")] NeuralNetwork,
}

impl Strategy {
    /// Higher value = more accurate (exact beats sampling).
    /// Used by the GUI to decide which strategy's probs to display.
    pub fn priority(self) -> u8 {
        match self {
            Strategy::MonteCarlo => 1,
            Strategy::ConstraintSearch => 2,
            #[cfg(feature = "neural")] Strategy::NeuralNetwork => 1,
        }
    }
}

/// Progress report sent through the channel by either strategy.
pub enum SimUpdate {
    /// Periodic snapshot while the simulation is still running.
    Progress {
        strategy: Strategy,
        attempts: usize,
        valid: usize,
        max_attempts: usize,
        memory_bytes: usize,
        probs: Vec<Vec<f64>>,
    },
    /// Sent once when the simulation finishes.
    Done {
        strategy: Strategy,
        attempts: usize,
        valid: usize,
        memory_bytes: usize,
        probs: Vec<Vec<f64>>,
    },
}
