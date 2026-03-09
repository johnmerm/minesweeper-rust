use crate::Minesweeper;

pub mod monte_carlo;
pub mod constraint_search;
pub mod fpga_bp;
pub use monte_carlo::MonteCarlo;
pub use constraint_search::ConstraintSearch;
pub use fpga_bp::FpgaBp;

pub trait ProbabilityStrategy {
    fn calculate(&self, game: &Minesweeper) -> Vec<Vec<f64>>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Strategy {
    MonteCarlo,
    ConstraintSearch,
    /// FPGA-ready Belief Propagation — approximate but tractable at large scale.
    FpgaBp,
}

impl Strategy {
    /// Higher value = more accurate (exact beats sampling).
    /// Used by the GUI to decide which strategy's probs to display.
    pub fn priority(self) -> u8 {
        match self {
            Strategy::MonteCarlo => 1,
            Strategy::FpgaBp => 2,
            Strategy::ConstraintSearch => 3,
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
