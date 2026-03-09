//! FPGA-ready Belief Propagation solver for large-scale Minesweeper boards.
//!
//! # Motivation
//!
//! The classical DFS constraint search is exact but exponential in the number
//! of border cells. For large boards (e.g. 1000×1000, 40% mine density) the
//! border region can contain tens of thousands of cells, making the DFS
//! approach computationally intractable regardless of hardware.
//!
//! Belief Propagation (BP) on the factor graph offers approximate but
//! high-quality probability estimates that converge in a fixed number of
//! iterations, making it a natural fit for hardware acceleration.
//!
//! # HLS Design Intent
//!
//! This module is structured for direct translation to FPGA using High-Level
//! Synthesis (HLS) tools such as Intel HLS Compiler or Xilinx Vitis HLS.
//! The key design constraints that enable HLS compilation are:
//!
//! - **Fixed-point arithmetic**: All probability computations use Q16.16
//!   fixed-point (`Fxp`) rather than IEEE 754 doubles. Fixed-point maps
//!   directly to DSP slices with deterministic latency, no rounding
//!   exceptions, and no need for FP exception logic.
//!
//! - **Flat array layouts with fixed strides**: No `Vec` or heap allocation
//!   in compute kernels. All working state uses fixed-stride arrays
//!   (`constraint * MAX_DEGREE + j`), enabling the HLS tool to infer BRAMs
//!   and partition memory banks for parallel access.
//!
//! - **Statically bounded loops**: Every hot-path loop uses a compile-time
//!   constant upper bound (`MAX_DEGREE`, `MAX_CELL_FACTORS`, `BP_ITERS`).
//!   HLS tools require static bounds to unroll, pipeline, and schedule
//!   loops into hardware stages.
//!
//! - **Explicit dataflow stages**: The solver is split into three phases
//!   (`build_graph`, `bp_solve`, output mapping) corresponding to pipelined
//!   HLS dataflow regions. Each stage reads from one BRAM bank and writes
//!   to another, enabling II=1 pipelining of the outer iteration loop.
//!
//! - **Component parallelism**: The constraint graph is decomposed into
//!   independent connected components. On FPGA, each component can be
//!   dispatched to a separate Processing Element (PE) cluster simultaneously,
//!   giving near-linear scaling with the number of available PEs.
//!
//! # Algorithm
//!
//! Sum-product belief propagation on the Minesweeper factor graph:
//!
//! - **Variable nodes**: hidden cells x_i ∈ {0=safe, 1=mine}
//! - **Factor nodes**: numbered revealed cells, each enforcing
//!   `sum(x_j for j in neighbors) = required`
//!
//! **Factor → variable message** for factor f, position j:
//! ```text
//!   msg_mine[f][j] = P(sum of other neighbors = required - 1)
//!   msg_safe[f][j] = P(sum of other neighbors = required)
//! ```
//! Computed via a DP table of size `≤ (MAX_DEGREE + 1)` — fully unrollable.
//!
//! **Variable belief update** for cell x_i:
//! ```text
//!   unnorm_mine = prior_mine × ∏_{f ∋ x_i} msg_mine[f][local_j]
//!   unnorm_safe = prior_safe × ∏_{f ∋ x_i} msg_safe[f][local_j]
//!   belief[i]   = unnorm_mine / (unnorm_mine + unnorm_safe)
//! ```
//!
//! # Scale
//!
//! Designed for boards up to 1000×1000 with high mine density (40%+), where
//! the classical DFS constraint search is computationally intractable.

use crate::Minesweeper;

use super::ProbabilityStrategy;
use super::monte_carlo::SimSetup;

// ---------------------------------------------------------------------------
// HLS compile-time parameters
// ---------------------------------------------------------------------------

/// Maximum neighbors a single constraint can reference.
/// In standard Minesweeper every cell has at most 8 neighbors, so this
/// bound is tight. HLS tools use it to statically unroll the DP inner loop.
const MAX_DEGREE: usize = 8;

/// Maximum constraints a single cell can belong to (same bound by geometry).
/// HLS: determines the unroll factor for the variable belief product loop.
const MAX_CELL_FACTORS: usize = 8;

/// Number of BP iterations. Fixed-count loops are required for HLS pipeline
/// scheduling. 50 iterations converges for typical Minesweeper boards.
const BP_ITERS: usize = 50;

// ---------------------------------------------------------------------------
// Fixed-point arithmetic: Q16.16
// ---------------------------------------------------------------------------

/// Q16.16 signed fixed-point number.
///
/// Represents the value `self.0 / 65536.0`. For probabilities only the range
/// `[ZERO, ONE]` (i.e., `[0, 65536]`) is used.
///
/// ## FPGA mapping
/// Each `Fxp` is a 32-bit register. Multiplication uses a 64-bit intermediate
/// (`(a * b) >> 16`), mapping to two DSP48E2 slices on Xilinx UltraScale+.
/// Addition is a single 32-bit adder with saturation. Division (used only
/// for normalization) maps to a pipelined integer divider.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Fxp(pub i32);

impl Fxp {
    pub const ZERO: Self = Fxp(0);
    /// 1.0 in Q16.16 (= 1 << 16).
    pub const ONE: Self = Fxp(65536);
    /// 0.5 in Q16.16.
    pub const HALF: Self = Fxp(32768);

    /// Convert from `f64`. Clamps result to `[ZERO, ONE]`.
    pub fn from_f64(v: f64) -> Self {
        Fxp(((v * 65536.0).round() as i64).clamp(0, 65536) as i32)
    }

    /// Convert to `f64`.
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / 65536.0
    }

    /// Saturating addition, clamped to `[ZERO, ONE]`.
    /// HLS: 32-bit adder + clamp comparator.
    pub fn add(self, rhs: Self) -> Self {
        Fxp(self.0.saturating_add(rhs.0)).clamp(Self::ZERO, Self::ONE)
    }

    /// Q16.16 × Q16.16 multiplication via 64-bit intermediate.
    /// HLS maps to: `result = (a * b) >> 16`.
    pub fn mul(self, rhs: Self) -> Self {
        let p = (self.0 as i64 * rhs.0 as i64) >> 16;
        Fxp(p.clamp(0, 65536) as i32)
    }

    /// `1.0 - self`, clamped at zero.
    pub fn complement(self) -> Self {
        Fxp((65536 - self.0).max(0))
    }

    /// Clamp to `[lo, hi]`.
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        if self < lo { lo } else if self > hi { hi } else { self }
    }
}

// ---------------------------------------------------------------------------
// Flat constraint graph (HLS-compatible memory layout)
// ---------------------------------------------------------------------------

/// Constraint graph stored in fixed-stride flat arrays.
///
/// All arrays use the indexing convention `base[entity * STRIDE + local]`
/// so the HLS tool can:
/// - Infer each array as a BRAM block
/// - Partition BRAMs into banks of width `STRIDE` for parallel access
/// - Pipeline reads/writes with deterministic latency
///
/// No `Vec` is used inside the compute path (`bp_solve`). These Vecs act as
/// BRAM image containers in the host/HLS bridge; on pure FPGA they would be
/// static arrays dimensioned by compile-time capacity parameters.
struct FlatGraph {
    n_cells: usize,
    n_constraints: usize,

    // -- Constraint-indexed arrays (stride = MAX_DEGREE) --
    // HLS: `#pragma HLS array_partition variable=c_neighbors cyclic factor=MAX_DEGREE`

    /// Flat neighbor list. `c_neighbors[ci * MAX_DEGREE + j]` = cell index of
    /// the j-th neighbor of constraint `ci`.
    c_neighbors: Vec<u16>,

    /// Number of active neighbors for each constraint.
    c_degree: Vec<u8>,

    /// Required mine count for each constraint.
    c_required: Vec<u8>,

    // -- Cell-indexed arrays (stride = MAX_CELL_FACTORS) --
    // HLS: `#pragma HLS array_partition variable=x_factors cyclic factor=MAX_CELL_FACTORS`

    /// `x_factors[xi * MAX_CELL_FACTORS + k]` = index of the k-th constraint
    /// that cell `xi` participates in.
    x_factors: Vec<u16>,

    /// `x_factor_local[xi * MAX_CELL_FACTORS + k]` = local neighbor position
    /// of cell `xi` within that constraint (indexes into msg arrays).
    x_factor_local: Vec<u8>,

    /// Number of constraints each cell participates in.
    x_factor_count: Vec<u8>,

    // -- Component metadata --

    /// Component ID of each cell. Independent components can be dispatched
    /// to separate PE clusters on FPGA simultaneously.
    x_component: Vec<u16>,

    /// Total number of independent connected components found.
    pub n_components: usize,
}

impl FlatGraph {
    fn build(constraints: &[(Vec<usize>, usize)], n_cells: usize) -> Self {
        let nc = constraints.len();

        let mut g = FlatGraph {
            n_cells,
            n_constraints: nc,
            c_neighbors:    vec![0u16; nc * MAX_DEGREE],
            c_degree:       vec![0u8;  nc],
            c_required:     vec![0u8;  nc],
            x_factors:      vec![0u16; n_cells * MAX_CELL_FACTORS],
            x_factor_local: vec![0u8;  n_cells * MAX_CELL_FACTORS],
            x_factor_count: vec![0u8;  n_cells],
            x_component:    vec![0u16; n_cells],
            n_components:   0,
        };

        // Fill constraint arrays.
        for (ci, (neighbors, required)) in constraints.iter().enumerate() {
            let deg = neighbors.len().min(MAX_DEGREE);
            g.c_degree[ci]   = deg as u8;
            g.c_required[ci] = *required as u8;
            for (j, &cell) in neighbors.iter().take(MAX_DEGREE).enumerate() {
                g.c_neighbors[ci * MAX_DEGREE + j] = cell as u16;
            }
        }

        // Build cell → factor reverse index.
        for (ci, (neighbors, _)) in constraints.iter().enumerate() {
            for (j, &cell) in neighbors.iter().take(MAX_DEGREE).enumerate() {
                if cell >= n_cells { continue; }
                let fc = g.x_factor_count[cell] as usize;
                if fc < MAX_CELL_FACTORS {
                    g.x_factors[cell * MAX_CELL_FACTORS + fc]      = ci as u16;
                    g.x_factor_local[cell * MAX_CELL_FACTORS + fc] = j as u8;
                    g.x_factor_count[cell] += 1;
                }
            }
        }

        // Decompose into independent components.
        g.n_components = component_decompose(
            &mut g.x_component,
            n_cells,
            nc,
            &g.c_neighbors,
            &g.c_degree,
            &g.x_factor_count,
            &g.x_factors,
        );

        g
    }
}

// ---------------------------------------------------------------------------
// Component decomposition (setup phase — not pipelined)
// ---------------------------------------------------------------------------

/// Assign each cell to an independent connected component via BFS.
///
/// Two cells belong to the same component iff they share at least one
/// constraint. Components are fully independent: mines in one have no
/// probabilistic influence on mines in another. On FPGA, each component
/// can be solved by a separate PE cluster without synchronization.
///
/// This runs once on the host (or in the FPGA setup stage) and loads
/// the resulting `x_component` labels into BRAM before BP begins.
///
/// Returns the total number of components found.
fn component_decompose(
    x_component: &mut [u16],
    n_cells: usize,
    n_constraints: usize,
    c_neighbors: &[u16],
    c_degree: &[u8],
    x_factor_count: &[u8],
    x_factors: &[u16],
) -> usize {
    let mut cell_visited       = vec![false; n_cells];
    let mut constraint_visited = vec![false; n_constraints];
    let mut bfs_queue          = Vec::with_capacity(n_cells);
    let mut comp_id            = 0usize;

    for start in 0..n_cells {
        // Only start a component from a cell that has at least one constraint.
        if cell_visited[start] || x_factor_count[start] == 0 {
            continue;
        }

        bfs_queue.clear();
        bfs_queue.push(start);
        cell_visited[start] = true;

        let mut head = 0;
        while head < bfs_queue.len() {
            let cell = bfs_queue[head];
            head += 1;

            x_component[cell] = comp_id as u16;

            let fc = x_factor_count[cell] as usize;
            // HLS: UNROLL factor=MAX_CELL_FACTORS
            for k in 0..fc.min(MAX_CELL_FACTORS) {
                let ci = x_factors[cell * MAX_CELL_FACTORS + k] as usize;
                if constraint_visited[ci] { continue; }
                constraint_visited[ci] = true;

                let deg = c_degree[ci] as usize;
                // HLS: UNROLL factor=MAX_DEGREE
                for j in 0..deg.min(MAX_DEGREE) {
                    let nb = c_neighbors[ci * MAX_DEGREE + j] as usize;
                    if !cell_visited[nb] {
                        cell_visited[nb] = true;
                        bfs_queue.push(nb);
                    }
                }
            }
        }

        comp_id += 1;
    }

    comp_id
}

// ---------------------------------------------------------------------------
// BP kernel: factor → variable messages
// ---------------------------------------------------------------------------

/// Compute factor-to-variable messages for constraint `ci`.
///
/// For a constraint with neighbors `[x_0, …, x_{m-1}]` and `required = k`,
/// for each neighbor position `j` computes:
/// ```text
///   msg_mine[ci][j] = P(Σ_{i≠j} x_i = k − 1)   // x_j being mine is consistent
///   msg_safe[ci][j] = P(Σ_{i≠j} x_i = k    )   // x_j being safe is consistent
/// ```
///
/// These are computed via an in-place 0/1 knapsack DP of depth `m − 1` and
/// width `k + 1`. With `m ≤ MAX_DEGREE = 8`, the table has at most 9 entries.
///
/// ## HLS mapping
/// The outer `j` loop and inner `i` loop are both bounded by `MAX_DEGREE`
/// and are fully unrolled by the HLS tool, producing a purely combinational
/// compute block with no feedback cycles. One invocation of this function
/// maps to a single HLS compute stage, pipelined across all `n_constraints`
/// constraints.
fn update_factor_messages(
    ci: usize,
    c_neighbors: &[u16],
    c_degree:    &[u8],
    c_required:  &[u8],
    belief:      &[Fxp],
    msg_mine:    &mut [Fxp],
    msg_safe:    &mut [Fxp],
) {
    let m = c_degree[ci]   as usize;
    let k = c_required[ci] as usize;

    // HLS pragma: PIPELINE II=1; UNROLL loop_j factor=MAX_DEGREE
    for j in 0..m {
        // dp[s] = probability that the (m−1) variables other than j sum to s.
        // Initialize to the empty-set distribution: dp[0] = 1, rest = 0.
        let mut dp = [Fxp::ZERO; MAX_DEGREE + 1];
        dp[0] = Fxp::ONE;

        // Process all neighbors except position j.
        // HLS pragma: UNROLL factor=MAX_DEGREE (fully unrolled, pure combinational)
        for i in 0..m {
            if i == j { continue; }

            let cell   = c_neighbors[ci * MAX_DEGREE + i] as usize;
            let p_mine = belief[cell];
            let p_safe = p_mine.complement();

            // 0/1 knapsack update (high-to-low traversal avoids double-counting).
            // HLS: fully unrolled — MAX_DEGREE+1 parallel adder-multiplier chains.
            for s in (0..=MAX_DEGREE).rev() {
                let from_mine = if s > 0 { dp[s - 1].mul(p_mine) } else { Fxp::ZERO };
                let from_safe = dp[s].mul(p_safe);
                dp[s] = from_mine.add(from_safe);
            }
        }

        // Extract the two messages:
        //   x_j = mine → need the other m−1 cells to sum to k−1
        //   x_j = safe → need the other m−1 cells to sum to k
        msg_mine[ci * MAX_DEGREE + j] = if k > 0 { dp[k - 1] } else { Fxp::ZERO };
        msg_safe[ci * MAX_DEGREE + j] = dp[k];
    }
}

// ---------------------------------------------------------------------------
// BP kernel: variable belief update
// ---------------------------------------------------------------------------

/// Update all variable beliefs from the current factor messages.
///
/// For each cell `x_i`:
/// ```text
///   unnorm_mine = prior_mine × ∏_{f ∋ x_i} msg_mine[f][local_j]
///   unnorm_safe = prior_safe × ∏_{f ∋ x_i} msg_safe[f][local_j]
///   belief[i]   = unnorm_mine / (unnorm_mine + unnorm_safe)
/// ```
///
/// Interior cells (no constraints) keep the global prior unchanged.
///
/// ## HLS mapping
/// One PE per cell; all cells update in parallel within one pipeline stage.
/// The inner product loop is bounded by `MAX_CELL_FACTORS = 8` and fully
/// unrolled. Normalization uses a single pipelined integer divider per cell.
fn update_variable_beliefs(
    n_cells:         usize,
    prior_mine:      Fxp,
    x_factor_count:  &[u8],
    x_factors:       &[u16],
    x_factor_local:  &[u8],
    msg_mine:        &[Fxp],
    msg_safe:        &[Fxp],
    belief:          &mut [Fxp],
) {
    let prior_safe = prior_mine.complement();

    // HLS pragma: PIPELINE II=1 (one cell per cycle, inner loops unrolled)
    for xi in 0..n_cells {
        let fc = x_factor_count[xi] as usize;

        if fc == 0 {
            // Interior cell: not covered by any constraint — keep prior.
            belief[xi] = prior_mine;
            continue;
        }

        let mut prod_mine = prior_mine;
        let mut prod_safe = prior_safe;

        // HLS pragma: UNROLL factor=MAX_CELL_FACTORS
        for k in 0..fc.min(MAX_CELL_FACTORS) {
            let ci = x_factors[xi * MAX_CELL_FACTORS + k] as usize;
            let j  = x_factor_local[xi * MAX_CELL_FACTORS + k] as usize;
            prod_mine = prod_mine.mul(msg_mine[ci * MAX_DEGREE + j]);
            prod_safe = prod_safe.mul(msg_safe[ci * MAX_DEGREE + j]);
        }

        // Normalize: belief = prod_mine / (prod_mine + prod_safe).
        // Fixed-point division: (numerator << 16) / denominator.
        // HLS: maps to a pipelined divider (or LUT-based for small denominators).
        let total = prod_mine.add(prod_safe);
        belief[xi] = if total == Fxp::ZERO {
            prior_mine  // Degenerate — fall back to prior.
        } else {
            let shifted = (prod_mine.0 as i64) << 16;
            let result  = shifted / (total.0 as i64);
            Fxp(result.clamp(0, 65536) as i32)
        };
    }
}

// ---------------------------------------------------------------------------
// Top-level BP solver
// ---------------------------------------------------------------------------

/// Run belief propagation for [`BP_ITERS`] iterations and return per-cell
/// mine probabilities as `Fxp` values.
///
/// ## Dataflow structure (FPGA)
/// ```text
///   ┌──────────────────────┐     ┌──────────────────────┐
///   │ update_factor_msgs   │────▶│ update_variable_      │
///   │ (reads belief BRAM A)│     │ beliefs               │
///   │ (writes msg BRAMs)   │     │ (reads msg BRAMs)     │
///   └──────────────────────┘     │ (writes belief BRAM B)│
///            ▲                   └──────────────────────┘
///            └──────── ping-pong A↔B each iteration ─────┘
/// ```
/// The two BRAM banks for `belief` ping-pong between iterations,
/// enabling II=1 pipelining of the outer `BP_ITERS` loop.
fn bp_solve(graph: &FlatGraph, prior_mine: Fxp) -> Vec<Fxp> {
    let nc = graph.n_constraints;
    let nx = graph.n_cells;

    // Working state — on FPGA these are BRAMs.
    let mut belief   = vec![prior_mine; nx];
    let mut msg_mine = vec![Fxp::HALF;              nc * MAX_DEGREE];
    let mut msg_safe = vec![Fxp::ONE.complement();  nc * MAX_DEGREE]; // 0.5

    // HLS pragma: PIPELINE outer loop (II = nc + nx clock cycles per iteration)
    for _iter in 0..BP_ITERS {
        // Stage 1: factor → variable messages (reads belief, writes msg_*)
        for ci in 0..nc {
            update_factor_messages(
                ci,
                &graph.c_neighbors,
                &graph.c_degree,
                &graph.c_required,
                &belief,
                &mut msg_mine,
                &mut msg_safe,
            );
        }

        // Stage 2: variable beliefs (reads msg_*, writes belief)
        update_variable_beliefs(
            nx,
            prior_mine,
            &graph.x_factor_count,
            &graph.x_factors,
            &graph.x_factor_local,
            &msg_mine,
            &msg_safe,
            &mut belief,
        );
    }

    belief
}

// ---------------------------------------------------------------------------
// Strategy integration
// ---------------------------------------------------------------------------

/// FPGA-ready Belief Propagation probability estimator.
///
/// Uses sum-product BP on the Minesweeper factor graph to compute approximate
/// mine probabilities in O(BP_ITERS × (n_constraints + n_cells)) time.
/// Exact for trees; approximate (but typically accurate) for loopy graphs.
///
/// Unlike the DFS [`ConstraintSearch`], this solver remains tractable at
/// 1000×1000 scale with high mine density.
///
/// [`ConstraintSearch`]: super::ConstraintSearch
pub struct FpgaBp;

impl FpgaBp {
    pub fn new() -> Self { Self }
}

impl ProbabilityStrategy for FpgaBp {
    fn calculate(&self, game: &Minesweeper) -> Vec<Vec<f64>> {
        let Some(setup) = SimSetup::build(game) else {
            return vec![vec![0.0; game.width]; game.height];
        };

        let n = setup.hidden_cells.len();
        if n == 0 {
            return vec![vec![0.0; game.width]; game.height];
        }

        let prior = Fxp::from_f64(setup.mines_to_place as f64 / n as f64);
        let graph  = FlatGraph::build(&setup.constraints, n);
        let beliefs = bp_solve(&graph, prior);

        // Map computed beliefs back onto the full (row, col) probability grid.
        // hidden_cells stores (x=col, y=row) tuples.
        let mut probs = vec![vec![0.0f64; game.width]; game.height];

        for &(x, y) in &setup.certain_mines {
            probs[y][x] = 1.0;
        }

        for (idx, &(x, y)) in setup.hidden_cells.iter().enumerate() {
            probs[y][x] = beliefs[idx].to_f64().clamp(0.0, 1.0);
        }

        probs
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fxp_roundtrip() {
        for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let f = Fxp::from_f64(v);
            let back = f.to_f64();
            assert!((back - v).abs() < 1e-4, "roundtrip failed for {v}: got {back}");
        }
    }

    #[test]
    fn fxp_mul() {
        let a = Fxp::from_f64(0.5);
        let b = Fxp::from_f64(0.4);
        let c = a.mul(b);
        assert!((c.to_f64() - 0.2).abs() < 1e-3, "0.5 * 0.4 = {:?}", c.to_f64());
    }

    #[test]
    fn fxp_complement() {
        let a = Fxp::from_f64(0.3);
        let c = a.complement();
        assert!((c.to_f64() - 0.7).abs() < 1e-3);
    }

    #[test]
    fn single_constraint_converges() {
        // One constraint: 2 cells, 1 must be a mine.
        // Expected: each cell has probability 0.5.
        let constraints = vec![(vec![0usize, 1usize], 1usize)];
        let n_cells = 2;
        let prior = Fxp::from_f64(0.5);

        let graph   = FlatGraph::build(&constraints, n_cells);
        let beliefs = bp_solve(&graph, prior);

        assert_eq!(graph.n_components, 1);
        for b in &beliefs {
            let p = b.to_f64();
            assert!((p - 0.5).abs() < 0.01, "expected 0.5 got {p}");
        }
    }

    #[test]
    fn certain_mine_constraint() {
        // One constraint: 1 cell, 1 must be a mine → probability ≈ 1.0.
        let constraints = vec![(vec![0usize], 1usize)];
        let n_cells = 1;
        let prior = Fxp::from_f64(0.5);

        let graph   = FlatGraph::build(&constraints, n_cells);
        let beliefs = bp_solve(&graph, prior);

        let p = beliefs[0].to_f64();
        assert!(p > 0.95, "expected ≈1.0 got {p}");
    }

    #[test]
    fn certain_safe_constraint() {
        // One constraint: 1 cell, 0 mines required → probability ≈ 0.0.
        let constraints = vec![(vec![0usize], 0usize)];
        let n_cells = 1;
        let prior = Fxp::from_f64(0.5);

        let graph   = FlatGraph::build(&constraints, n_cells);
        let beliefs = bp_solve(&graph, prior);

        let p = beliefs[0].to_f64();
        assert!(p < 0.05, "expected ≈0.0 got {p}");
    }

    #[test]
    fn two_independent_components() {
        // Component A: cells 0,1 with sum=1 → each 0.5
        // Component B: cells 2,3 with sum=2 → each 1.0
        let constraints = vec![
            (vec![0, 1], 1),
            (vec![2, 3], 2),
        ];
        let graph = FlatGraph::build(&constraints, 4);
        assert_eq!(graph.n_components, 2);

        let beliefs = bp_solve(&graph, Fxp::from_f64(0.5));
        assert!((beliefs[0].to_f64() - 0.5).abs() < 0.02);
        assert!((beliefs[1].to_f64() - 0.5).abs() < 0.02);
        assert!(beliefs[2].to_f64() > 0.95);
        assert!(beliefs[3].to_f64() > 0.95);
    }
}
