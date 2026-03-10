use crate::Minesweeper;

pub mod monte_carlo;
pub mod constraint_search;
#[cfg(feature = "neural")] pub mod neural;
pub use monte_carlo::MonteCarlo;
pub use constraint_search::ConstraintSearch;
#[cfg(feature = "neural")] pub use neural::NeuralNetwork;

pub trait ProbabilityStrategy {
    fn calculate(&self, game: &Minesweeper) -> Vec<Vec<f64>>;
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
