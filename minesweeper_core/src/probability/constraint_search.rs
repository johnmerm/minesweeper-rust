use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::mpsc::Sender;

use crate::Minesweeper;

use super::{ProbabilityStrategy, SimUpdate, Strategy};
use super::components::{combine, decompose, signature, ComponentSolution, SolutionCache};
use super::monte_carlo::{build_probs, mc_memory_estimate, SimSetup};
use super::MonteCarlo;

/// Exact mine probability estimation using depth-first constraint enumeration.
///
/// # High-level idea
///
/// Every visible numbered cell gives us a *constraint*: "exactly N of my hidden
/// neighbours are mines."  Instead of sampling random boards, we walk every
/// assignment of mines/safes to hidden cells that simultaneously satisfies all
/// constraints.
///
/// # Two kinds of hidden cells
///
/// ```text
///  ┌───┬───┬───┐
///  │ 2 │ ? │ ? │  ← "border" cells (? adjacent to a numbered cell)
///  ├───┼───┼───┤     These are directly constrained.
///  │ ? │ ? │ ? │  ← "interior" cells (? not adjacent to any numbered cell)
///  └───┴───┴───┘     No constraint tells us exactly which are mines.
/// ```
///
/// For border cells we enumerate every valid assignment explicitly.
/// For interior cells we know only the total count of mines left over after
/// the border is fixed, so we treat them analytically (uniform distribution).
///
/// # Search algorithm
///
/// The border is first split into independent groups — see [`super::components`]
/// — and each is searched on its own, because enumerating them together walks the
/// product of their solution counts rather than the sum.
///
/// Within a group we process constraints one at a time (depth = constraint index).
/// At each level we look at the current constraint's unassigned neighbours and
/// pick which `needed` of the `m` unassigned cells are mines — that's C(m, needed)
/// choices.  We fix them, recurse to the next constraint, then backtrack.
///
/// Because earlier constraints already fixed some cells shared with later ones,
/// the branching factor shrinks rapidly → the tree is tiny compared with brute-force.
///
/// Each *leaf* (all of the group's constraints satisfied) is recorded against the
/// number of mines it used, and nothing else: how likely that leaf is depends on
/// what the rest of the board does, which is not known here. Combining the groups,
/// and weighting by the ways the interior can hold whatever mines are left over,
/// happens in [`super::components::combine`].
pub struct ConstraintSearch {
    /// Search nodes to visit before giving up on being exact.
    ///
    /// The search is exponential in the size of the border, and on a dense board
    /// a single position can hold billions of consistent layouts — one 30x30/250
    /// position took over two minutes, which is indistinguishable from a hang.
    /// Past this many nodes the search stops and reports no result, so the caller
    /// falls back to sampling instead of waiting.
    ///
    /// Set it to `usize::MAX` for an unbounded, always-exact search when latency
    /// does not matter (offline analysis, training-data generation).
    pub max_nodes: usize,
    /// Component solutions kept between boards.
    ///
    /// Reuse depends entirely on how long this instance lives: a caller that
    /// builds a `ConstraintSearch` per move gets the previous behaviour with an
    /// empty map, and one that keeps it across moves gets the cache. That is the
    /// whole opt-in — there is no flag.
    cache: RefCell<SolutionCache>,
}

/// Node budget that keeps a single solve inside a comfortable interactive
/// frame on the boards measured here, while still finishing the vast majority
/// of positions exactly.
const DEFAULT_MAX_NODES: usize = 1_000_000;

impl ConstraintSearch {
    pub fn new() -> Self {
        Self {
            max_nodes: DEFAULT_MAX_NODES,
            cache: RefCell::default(),
        }
    }

    /// An unbounded search: always exact, however long it takes.
    pub fn exhaustive() -> Self {
        Self {
            max_nodes: usize::MAX,
            cache: RefCell::default(),
        }
    }

    /// Component solves served from the cache, and solves that had to be done.
    ///
    /// Both are zero for an instance that is not kept between moves.
    pub fn cache_counts(&self) -> (u32, u32) {
        self.cache.borrow().counts()
    }

    /// Solve the board and report the result through `tx`.
    ///
    /// Unlike the sampling strategy this sends no intermediate snapshots. A
    /// half-finished exact search is not a weaker answer, it is a wrong one: the
    /// walk has covered only a lexicographic prefix of the layouts, so a cell can
    /// read 0% purely because its subtree has not been visited, and every caller
    /// treats 0% as proof that a cell is safe to open. The node budget bounds the
    /// work instead.
    pub fn calculate_with_progress(&self, game: &Minesweeper, tx: Sender<SimUpdate>) {
        let empty = || vec![vec![0.0; game.width]; game.height];

        let Some(setup) = SimSetup::build(game) else {
            // No hidden cells reachable (game not started yet, already won/lost, …).
            let _ = tx.send(SimUpdate::Done {
                strategy: Strategy::ConstraintSearch,
                attempts: 0,
                valid: 0,
                memory_bytes: 0,
                probs: empty(),
            });
            return;
        };

        let memory_bytes = cs_memory_estimate(&setup);
        let outcome = self.solve(&setup);

        let (probs, valid, attempts) = match outcome {
            Some(solved) => (
                build_probs(&solved.probabilities, 1.0, &setup, game.width, game.height),
                solved.layouts,
                solved.nodes,
            ),
            // No usable answer — the caller falls back to sampling. Reporting zero
            // layouts is what tells it to; a grid of zeros would read as "every
            // cell is provably safe".
            None => (empty(), 0, 0),
        };

        let _ = tx.send(SimUpdate::Done {
            strategy: Strategy::ConstraintSearch,
            attempts,
            valid,
            memory_bytes,
            probs,
        });
    }

    /// Solve every component and combine them, or `None` if no trustworthy answer
    /// came out.
    fn solve(&self, setup: &SimSetup) -> Option<Solved> {
        let components = decompose(setup);

        // Cells no number speaks about. They are not searched: their probability
        // follows from how many mines the components leave over.
        let constrained: HashSet<usize> = components
            .iter()
            .flat_map(|c| c.cells.iter().copied())
            .collect();
        let interior: Vec<usize> = (0..setup.hidden_cells.len())
            .filter(|i| !constrained.contains(i))
            .collect();

        // The budget is for the whole board, so components share it. A component
        // that runs out makes the entire answer untrustworthy: its own numbers are
        // biased, and they feed every other cell through the combination.
        let mut remaining = self.max_nodes;
        let mut solutions: Vec<Rc<ComponentSolution>> = Vec::with_capacity(components.len());
        let mut layouts = 0usize;
        let mut nodes = 0usize;

        for component in &components {
            // Most of the board is untouched by any one move, so most components
            // come back exactly as they were and their answers can be reused.
            let key = signature(component, &setup.hidden_cells);
            if let Some(solution) = self.cache.borrow_mut().get(&key) {
                solutions.push(solution);
                continue;
            }

            // Bounded by the component's own size and nothing else. Capping it at
            // the board's remaining mines would be free today, but it would make
            // the result depend on the rest of the board — and its independence is
            // the whole reason components can be solved separately, and the reason
            // one can be cached across moves at all. Layouts using more mines than
            // the board has left are discarded in the combination, where the rest
            // of the board is actually known.
            let mut dfs = Dfs::new(
                component.cells.len(),
                &component.constraints,
                component.cells.len(),
                remaining,
            );
            dfs.run(0);
            if dfs.exhausted {
                return None;
            }
            remaining = remaining.saturating_sub(dfs.nodes);
            nodes += dfs.nodes;
            layouts += dfs.valid_count as usize;

            let solution = Rc::new(ComponentSolution {
                ways: dfs.ways,
                cell_ways: dfs.cell_ways,
            });
            self.cache.borrow_mut().insert(key, Rc::clone(&solution));
            solutions.push(solution);
        }

        let probabilities = combine(setup, &components, &solutions, &interior)?;
        Some(Solved {
            probabilities,
            // With no constrained cells at all there is still exactly one layout:
            // the empty one. Reporting zero would read as "no answer".
            layouts: layouts.max(1),
            nodes,
        })
    }
}

/// A trustworthy answer: one probability per uncertain hidden cell.
struct Solved {
    probabilities: Vec<f64>,
    layouts: usize,
    nodes: usize,
}

impl ProbabilityStrategy for ConstraintSearch {
    /// Synchronous version used by the CLI / web — runs to completion and returns probs.
    fn calculate(&self, game: &Minesweeper) -> Vec<Vec<f64>> {
        let Some(setup) = SimSetup::build(game) else {
            return vec![vec![0.0; game.width]; game.height];
        };
        match self.solve(&setup) {
            Some(solved) => {
                build_probs(&solved.probabilities, 1.0, &setup, game.width, game.height)
            }
            // This signature has no way to say "no answer", and zeros would be read
            // as proof of safety, so hand back a sampled estimate instead.
            None => MonteCarlo::new().calculate(game),
        }
    }
}

/// Depth-first search over one component's constraint tree.
///
/// `'a` ties the struct to the lifetime of the constraint slice.
/// `F` is the progress callback type.
///
/// It answers a question about that component alone: for each `k`, how many of
/// its layouts use exactly `k` mines, and in how many of those is a given cell
/// one. The rest of the board never enters, which is what lets the caller solve
/// components separately and combine them afterwards.
struct Dfs<'a> {
    /// Slice of (component-local cell indices, required-mine-count) pairs, one
    /// per visible numbered cell bearing on this component. Processed
    /// left-to-right, so depth = index into this slice.
    constraints: &'a [(Vec<usize>, usize)],
    /// Upper bound on mines in this component: no more than it has cells, and no
    /// more than the board has left to place.
    max_mines: usize,
    /// Current partial assignment.  `None` = not yet decided, `Some(true)` = mine,
    /// `Some(false)` = safe.  Indexed by component-local cell index.
    assignment: Vec<Option<bool>>,
    /// `ways[k]` — layouts found so far that use exactly k mines.
    ways: Vec<f64>,
    /// `cell_ways[c][k]` — of those, the ones where local cell c is a mine.
    cell_ways: Vec<Vec<f64>>,
    /// Mines currently placed. Maintained as the search walks rather than
    /// recounted at each leaf, which was O(cells) per leaf and dominated the
    /// whole search on a large board.
    mines_placed: usize,
    /// The cells behind that count, newest last, so a leaf can credit exactly the
    /// cells it placed mines on instead of scanning every cell.
    mine_cells: Vec<usize>,
    /// Number of valid leaves (constraints fully satisfied).
    valid_count: u32,
    /// Set when the budget runs out; causes all recursion to unwind.
    aborted: bool,
    /// Nodes visited so far, against `max_nodes`.
    nodes: usize,
    /// Budget; see [`ConstraintSearch::max_nodes`].
    max_nodes: usize,
    /// Set when the budget ran out. The partial numbers are then worthless — a
    /// half-finished depth-first walk has only covered a lexicographic prefix of
    /// the layouts, so a cell can read 0% purely because its subtree was never
    /// visited, which would be read as "provably safe". Callers must discard the
    /// result rather than display it.
    exhausted: bool,
}

impl<'a> Dfs<'a> {
    fn new(
        cells: usize,
        constraints: &'a [(Vec<usize>, usize)],
        max_mines: usize,
        max_nodes: usize,
    ) -> Self {
        Self {
            constraints,
            max_mines,
            assignment: vec![None; cells],
            ways: vec![0.0; max_mines + 1],
            cell_ways: vec![vec![0.0; max_mines + 1]; cells],
            mines_placed: 0,
            mine_cells: Vec::new(),
            valid_count: 0,
            aborted: false,
            nodes: 0,
            max_nodes,
            exhausted: false,
        }
    }

    /// Recursively assign mines/safes to satisfy `constraints[constraint_idx]`,
    /// then call `run(constraint_idx + 1)`.  Backtracks when done.
    fn run(&mut self, constraint_idx: usize) {
        if self.aborted {
            return;
        }

        // Budget check at every node, so a deep subtree that never reaches a leaf
        // cannot run away — the old cancellation only fired every 500th *valid*
        // leaf, which such a subtree never produces.
        self.nodes += 1;
        if self.nodes > self.max_nodes {
            self.aborted = true;
            self.exhausted = true;
            return;
        }

        // Base case: all constraints satisfied → score this leaf.
        if constraint_idx == self.constraints.len() {
            self.process_leaf();
            return;
        }

        // Clone the constraint data to avoid holding a borrow of `self.constraints`
        // while we mutate `self.assignment` during recursion.
        let (neighbors, required) = {
            let c = &self.constraints[constraint_idx];
            (c.0.clone(), c.1)
        };

        // How many of this constraint's neighbours are already marked as mines
        // by a previous constraint that shares cells with this one?
        let already_mines: usize = neighbors
            .iter()
            .filter(|&&i| self.assignment[i] == Some(true))
            .count();

        // Pruning: if we've already exceeded the required count, this branch is invalid.
        if already_mines > required {
            return;
        }

        // Cells in this constraint not yet assigned by an earlier constraint.
        let unassigned: Vec<usize> = neighbors
            .iter()
            .filter(|&&i| self.assignment[i].is_none())
            .copied()
            .collect();

        // How many additional mines we still need to place from the unassigned cells.
        let needed = required - already_mines;
        let m = unassigned.len();

        // Pruning: can't satisfy the constraint if there aren't enough free cells.
        if needed > m {
            return;
        }

        // Fast path: all unassigned cells are already decided (needed == 0 and m == 0,
        // or m > 0 but needed == 0 means they must all be safe).
        if m == 0 {
            // Every cell in this constraint is already fixed; constraint is satisfied.
            self.run(constraint_idx + 1);
            return;
        }

        if needed == 0 {
            // Zero additional mines needed → every unassigned neighbour must be safe.
            for &cell in &unassigned {
                self.set_cell(cell, false);
            }
            self.run(constraint_idx + 1);
            self.clear_cells(&unassigned);
            return;
        }

        if needed == m {
            // All unassigned neighbours must be mines (no choice).
            for &cell in &unassigned {
                self.set_cell(cell, true);
            }
            if self.mines_placed <= self.max_mines {
                self.run(constraint_idx + 1);
            }
            self.clear_cells(&unassigned);
            return;
        }

        // General case: choose `needed` mines out of `m` unassigned cells.
        // We iterate over all C(m, needed) subsets in lexicographic order.
        //
        // `combo` holds the *positions* (0..m) of the chosen mines.
        // Initially [0, 1, 2, …, needed-1] — the first subset.
        let mut combo: Vec<usize> = (0..needed).collect();
        // Allocated once for the whole loop rather than per combination.
        let mut is_mine_pos = vec![false; m];
        loop {
            if self.aborted {
                break;
            }

            // Apply this combination: mark selected positions as mines, rest as safe.
            is_mine_pos.iter_mut().for_each(|slot| *slot = false);
            for &ci in &combo {
                is_mine_pos[ci] = true;
            }
            for (j, &cell) in unassigned.iter().enumerate() {
                self.set_cell(cell, is_mine_pos[j]);
            }

            // Global budget: a branch that has already placed more mines than the
            // board holds cannot lead anywhere. Without this the search descends
            // through every remaining constraint before noticing at the leaf.
            if self.mines_placed <= self.max_mines {
                self.run(constraint_idx + 1);
            }

            // Backtrack: clear all assignments made by this constraint level so
            // the next combination starts from a clean slate.
            self.clear_cells(&unassigned);

            // Advance `combo` to the next combination in lexicographic order.
            // Find the rightmost position that can still be incremented.
            //
            // Example with m=5, needed=3, combo=[1,3,4]:
            //   i starts at 3 (== needed).
            //   combo[2]=4 == 5-3+2=4 → at max, decrement i to 2.
            //   combo[1]=3 == 5-3+1=3 → at max, decrement i to 1.
            //   combo[0]=1 != 5-3+0=2 → stop, i=1.
            //   Increment combo[0] to 2, fill rest: combo = [2,3,4].
            let mut i = needed;
            while i > 0 && combo[i - 1] == m - needed + i - 1 {
                i -= 1;
            }
            if i == 0 {
                break; // All combinations exhausted.
            }
            combo[i - 1] += 1;
            for j in i..needed {
                combo[j] = combo[j - 1] + 1;
            }
        }
    }

    /// Record one assignment, keeping the running mine count in step.
    fn set_cell(&mut self, cell: usize, mine: bool) {
        self.assignment[cell] = Some(mine);
        if mine {
            self.mines_placed += 1;
            self.mine_cells.push(cell);
        }
    }

    /// Undo the assignments made for one constraint level.
    ///
    /// Cleared in reverse so `mine_cells` unwinds exactly as it was built — it is
    /// a stack, and the leaf handler depends on it holding precisely the cells
    /// currently assigned as mines.
    fn clear_cells(&mut self, cells: &[usize]) {
        for &cell in cells.iter().rev() {
            if self.assignment[cell] == Some(true) {
                self.mines_placed -= 1;
                debug_assert_eq!(self.mine_cells.last(), Some(&cell));
                self.mine_cells.pop();
            }
            self.assignment[cell] = None;
        }
    }

    /// Called when all constraints are satisfied (we're at a leaf of the search tree).
    ///
    /// Records the layout against the number of mines it used. This runs once per
    /// valid layout — millions of times on a dense board — so it is O(mines
    /// placed) and never O(cells).
    fn process_leaf(&mut self) {
        let k = self.mines_placed;
        if k > self.max_mines {
            return; // More mines than the board has left; not a real layout.
        }

        self.ways[k] += 1.0;
        for i in 0..self.mine_cells.len() {
            let cell = self.mine_cells[i];
            self.cell_ways[cell][k] += 1.0;
        }
        self.valid_count += 1;
    }
}

// ---------------------------------------------------------------------------
// Memory estimation
// ---------------------------------------------------------------------------

/// Rough heap estimate for the Constraint Search strategy's working set.
///
/// Accounts for SimSetup storage (shared with MC) plus the DFS-specific
/// structures: assignment vector, mine-count accumulator, and call-stack
/// frame cost (combo + unassigned + is_mine_pos vectors per depth level).
fn cs_memory_estimate(setup: &SimSetup) -> usize {
    let n = setup.hidden_cells.len();
    let total_neighbors: usize = setup.constraints.iter().map(|(ns, _)| ns.len()).sum();
    let c = setup.constraints.len();

    // SimSetup heap (same formula as in mc_memory_estimate)
    let setup_heap = mc_memory_estimate(setup);

    // The dominant term is the per-group tables: `cell_ways` holds one row of
    // `mines + 1` counts per cell. Estimated against the whole border rather than
    // the true split, which this function cannot see — an upper bound, since one
    // group of n cells costs more than the same n cells split across two.
    let tables = n * (n + 1) * 8 + (n + 1) * 8;

    // DFS working set: the call stack goes `c` levels deep (one per constraint).
    // At each level we allocate three temporary vectors of size ≈ avg_unassigned.
    let avg_unassigned = total_neighbors.checked_div(c).unwrap_or(0);
    let stack_frame = avg_unassigned * 8  // combo: Vec<usize>
        + avg_unassigned * 8             // unassigned: Vec<usize>
        + avg_unassigned                 // is_mine_pos: Vec<bool>
        + 64;                            // Dfs struct overhead per frame
    let dfs_stack = c * stack_frame;

    let working = n          // assignment: Vec<Option<bool>> (1 byte each)
        + tables
        + dfs_stack;

    setup_heap + working
}