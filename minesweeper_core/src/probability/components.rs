//! Splitting the constraint graph into independent sub-problems, and putting
//! their answers back together.
//!
//! # Why
//!
//! Two hidden cells are related only if some visible number counts them both,
//! directly or through a chain of other cells. The border therefore falls into
//! groups that share no cell at all, and a mine layout for one group constrains
//! a layout for another only through the single global fact that the whole board
//! holds a fixed number of mines.
//!
//! Enumerating the border as one problem walks the *Cartesian product* of those
//! groups' solutions. Solving each group on its own and combining the results
//! turns that product into a sum:
//!
//! ```text
//!   one problem:      solutions(A) x solutions(B) x solutions(C)
//!   three problems:   solutions(A) + solutions(B) + solutions(C)
//! ```
//!
//! On a large sparse board this is the difference between intractable and
//! instant — a 50x50 board with 150 mines splits into around 27 groups per move.
//!
//! # How the pieces are combined
//!
//! A group's answer cannot be a probability, because how likely its layouts are
//! depends on how many mines the *rest* of the board takes. What is independent
//! of the rest — and this is the property the whole scheme rests on — is the
//! count of layouts per mine count:
//!
//! * `ways[k]`           — layouts of this group that use exactly `k` mines
//! * `cell_ways[c][k]`   — how many of those have cell `c` as a mine
//!
//! Neither depends on the total mine count, or on any other group. They are a
//! complete, reusable description of the group.
//!
//! Combining them is a convolution: the number of ways the whole border uses `b`
//! mines is the sum over every way of splitting `b` between the groups. Fold in
//! `C(interior, mines_left)` for the unconstrained cells and the probabilities
//! fall out.

use std::collections::HashMap;
use std::rc::Rc;

use super::monte_carlo::SimSetup;

/// One independent sub-problem: cells bound together by shared constraints, and
/// sharing no cell with any other component.
pub(crate) struct Component {
    /// Indices into `SimSetup::hidden_cells`, ascending.
    pub(crate) cells: Vec<usize>,
    /// The constraints over those cells, rebased to `0..cells.len()`.
    pub(crate) constraints: Vec<(Vec<usize>, usize)>,
}

/// A component's solution, in the form that makes it independent of everything
/// else on the board: counts indexed by how many mines the component uses.
pub(crate) struct ComponentSolution {
    /// `ways[k]` — layouts using exactly `k` mines. Length `cells.len() + 1`.
    pub(crate) ways: Vec<f64>,
    /// `cell_ways[c][k]` — layouts using `k` mines in which local cell `c` is one.
    pub(crate) cell_ways: Vec<Vec<f64>>,
}

impl ComponentSolution {
    /// Rough heap cost, for the cache's size accounting.
    fn weight(&self) -> usize {
        self.ways.len() * 8 + self.cell_ways.iter().map(|row| row.len() * 8 + 24).sum::<usize>()
    }
}

/// What a component is, independently of the board it came from: which squares
/// it covers and what the numbers say about them.
///
/// Two components with the same signature have the same solution — not merely a
/// similar one — because [`ComponentSolution`] is a function of exactly these
/// inputs and nothing else. That is what lets solutions be reused across moves,
/// and why the cache needs no invalidation: an entry cannot go stale, since a
/// board that changes a component changes its signature too.
///
/// Board coordinates, deliberately. Hidden-cell indices are renumbered by
/// `SimSetup::build` on every move as cells are opened and proved, so a key built
/// from them would miss every time — or worse, collide between different squares.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct Signature {
    /// The component's squares, in its own local order.
    cells: Vec<(u16, u16)>,
    /// Constraints as (local cell indices, mines required), sorted so that the
    /// same set of numbers always produces the same key.
    constraints: Vec<(Vec<u16>, u16)>,
}

impl Signature {
    /// Rough heap cost of keeping this key, for the cache's size accounting.
    fn weight(&self) -> usize {
        self.cells.len() * 4
            + self
                .constraints
                .iter()
                .map(|(cells, _)| cells.len() * 2 + 8)
                .sum::<usize>()
    }
}

/// Describe a component in board terms.
///
/// `component.cells` is ascending in hidden-cell index, and `SimSetup::build`
/// numbers hidden cells in row-major order, so the local indices are already
/// canonical: the same squares always produce the same local numbering.
pub(crate) fn signature(component: &Component, hidden_cells: &[(usize, usize)]) -> Signature {
    let cells = component
        .cells
        .iter()
        .map(|&i| {
            let (x, y) = hidden_cells[i];
            (x as u16, y as u16)
        })
        .collect();

    let mut constraints: Vec<(Vec<u16>, u16)> = component
        .constraints
        .iter()
        .map(|(cells, required)| {
            let mut cells: Vec<u16> = cells.iter().map(|&c| c as u16).collect();
            cells.sort_unstable();
            (cells, *required as u16)
        })
        .collect();
    constraints.sort();

    Signature { cells, constraints }
}

/// Component solutions kept from one move to the next.
///
/// Opening a square changes the numbers around it and nothing else, so most of
/// the board's components come back identical and their answers can be reused.
/// Entries never need invalidating — see [`Signature`] — so the only reason to
/// drop one is to bound memory.
#[derive(Default)]
pub struct SolutionCache {
    entries: HashMap<Signature, Rc<ComponentSolution>>,
    /// Rough heap bytes held, so a long game cannot grow this without limit.
    bytes: usize,
    hits: u32,
    misses: u32,
}

impl SolutionCache {
    /// Roughly 8 MB of solutions. Past this the cache is emptied rather than
    /// evicted one by one: entries stop being useful once the board has moved on,
    /// so tracking recency would cost more than it saves.
    const MAX_BYTES: usize = 8 << 20;

    pub(crate) fn get(&mut self, signature: &Signature) -> Option<Rc<ComponentSolution>> {
        match self.entries.get(signature) {
            Some(solution) => {
                self.hits += 1;
                Some(Rc::clone(solution))
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    pub(crate) fn insert(&mut self, signature: Signature, solution: Rc<ComponentSolution>) {
        let cost = signature.weight() + solution.weight();
        if self.bytes + cost > Self::MAX_BYTES {
            self.entries.clear();
            self.bytes = 0;
        }
        if self.entries.insert(signature, solution).is_none() {
            self.bytes += cost;
        }
    }

    /// How many component solves have been served from the cache, and how many
    /// had to be computed. Reported by the front-ends; also how the tests check
    /// that reuse is actually happening.
    pub fn counts(&self) -> (u32, u32) {
        (self.hits, self.misses)
    }
}

/// Partition the constrained cells into connected components.
///
/// Cells in no constraint are absent from the result: they are the *interior*,
/// which no number speaks about, and which is handled analytically rather than
/// by search.
pub(crate) fn decompose(setup: &SimSetup) -> Vec<Component> {
    let n = setup.hidden_cells.len();
    let mut parent: Vec<usize> = (0..n).collect();

    // `SimSetup::build` drops constraints with no hidden neighbours, but that is
    // its invariant, not ours, and the grouping below indexes `cells[0]`.
    let constraints = || setup.constraints.iter().filter(|(cells, _)| !cells.is_empty());

    // Union every pair of cells that appear in the same constraint.
    for (cells, _) in constraints() {
        for &cell in cells.iter().skip(1) {
            union(&mut parent, cells[0], cell);
        }
    }

    // Group the constraints by the root of the cells they touch.
    let mut by_root: HashMap<usize, Component> = HashMap::new();
    for (cells, required) in constraints() {
        let root = find(&mut parent, cells[0]);
        let component = by_root.entry(root).or_insert_with(|| Component {
            cells: Vec::new(),
            constraints: Vec::new(),
        });
        component.constraints.push((cells.clone(), *required));
    }

    // Collect each component's cells, then rebase its constraints onto them.
    for component in by_root.values_mut() {
        let mut cells: Vec<usize> = component
            .constraints
            .iter()
            .flat_map(|(cells, _)| cells.iter().copied())
            .collect();
        cells.sort_unstable();
        cells.dedup();

        let local: HashMap<usize, usize> =
            cells.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        for (constraint_cells, _) in component.constraints.iter_mut() {
            for cell in constraint_cells.iter_mut() {
                *cell = local[cell];
            }
        }
        component.cells = cells;
    }

    let mut components: Vec<Component> = by_root.into_values().collect();
    // Deterministic order: the same board must always produce the same answer,
    // and a stable order keeps results reproducible across runs.
    components.sort_by(|a, b| a.cells.first().cmp(&b.cells.first()));
    components
}

fn find(parent: &mut Vec<usize>, mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]]; // path halving
        i = parent[i];
    }
    i
}

fn union(parent: &mut Vec<usize>, a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent[rb] = ra;
    }
}

/// Per-cell mine probabilities for every uncertain hidden cell, given each
/// component's solution.
///
/// `probabilities[i]` corresponds to `setup.hidden_cells[i]`.
///
/// The global mine count is the only thing tying the components together, so it
/// enters here and nowhere else: for each way of splitting the mines between the
/// components, the unconstrained interior takes whatever is left, in
/// `C(interior, left)` ways.
pub(crate) fn combine(
    setup: &SimSetup,
    components: &[Component],
    solutions: &[Rc<ComponentSolution>],
    interior: &[usize],
) -> Option<Vec<f64>> {
    let mines_total = setup.mines_to_place;
    let interior_len = interior.len();
    let border_len: usize = components.iter().map(|c| c.cells.len()).sum();

    // Ways to put the leftover mines in the interior. Rescaled, because the true
    // values overflow f64 on any sizeable board — see `scaled_binomials`.
    let interior_ways = scaled_binomials(
        interior_len,
        mines_total.saturating_sub(border_len),
        mines_total.min(interior_len),
    );

    // Prefix[j] is the convolution of components 0..j, suffix[j] of j+1.., so
    // `rest` for component j is prefix[j] * suffix[j] — the distribution of every
    // other component combined. Computed this way rather than by dividing the
    // total, which would be numerically hopeless where a component contributes a
    // zero.
    let prefix = running_convolution(solutions.iter().map(|s| &s.ways));
    let suffix = {
        let mut s = running_convolution(solutions.iter().rev().map(|s| &s.ways));
        s.reverse();
        s
    };

    let total_ways = prefix.last().cloned().unwrap_or_else(|| vec![1.0]);
    let weight_of = |border_mines: usize| -> f64 {
        match mines_total.checked_sub(border_mines) {
            Some(left) if left <= interior_len => interior_ways[left],
            _ => 0.0,
        }
    };

    let total: f64 = total_ways
        .iter()
        .enumerate()
        .map(|(b, &ways)| ways * weight_of(b))
        .sum();
    if !(total > 0.0) || !total.is_finite() {
        return None;
    }

    let mut probabilities = vec![0.0f64; setup.hidden_cells.len()];

    for (j, (component, solution)) in components.iter().zip(solutions).enumerate() {
        // Everything except this component.
        let rest = convolve(&prefix[j], &suffix[j + 1]);

        // For a cell used by `k` of this component's mines, the rest of the board
        // may hold any `b`, and the interior takes what remains.
        let mut share = vec![0.0f64; solution.ways.len()];
        for (k, share_k) in share.iter_mut().enumerate() {
            *share_k = rest
                .iter()
                .enumerate()
                .map(|(b, &ways)| ways * weight_of(b + k))
                .sum();
        }

        for (local, &cell) in component.cells.iter().enumerate() {
            let cell_ways = &solution.cell_ways[local];
            let numerator: f64 = cell_ways
                .iter()
                .zip(&share)
                .map(|(&ways, &share_k)| ways * share_k)
                .sum();

            // Whether a cell is *certain* is decided from the layout counts, which
            // are exact integers, and never from the ratio. Weights are rescaled
            // and can underflow to zero in the tails, and a zero read off the ratio
            // would claim the cell is provably safe — proof every caller acts on by
            // opening it. The counts cannot lie that way.
            let never_a_mine = cell_ways.iter().all(|&ways| ways == 0.0);
            let always_a_mine = cell_ways
                .iter()
                .zip(&solution.ways)
                .all(|(&mine, &total)| mine == total);

            probabilities[cell] = settle(numerator / total, never_a_mine, always_a_mine)?;
        }
    }

    // Every interior cell is alike: across the layouts that leave `left` mines
    // for the interior, each of its cells is a mine in `left / interior_len` of
    // them.
    if interior_len > 0 {
        let numerator: f64 = total_ways
            .iter()
            .enumerate()
            .map(|(b, &ways)| {
                let left = match mines_total.checked_sub(b) {
                    Some(left) if left <= interior_len => left,
                    _ => return 0.0,
                };
                ways * weight_of(b) * left as f64 / interior_len as f64
            })
            .sum();
        // The interior is never certain either way: no number speaks about it, so
        // it holds a mine in some layouts and not others unless the board is
        // entirely decided, which `mines_total` covers below.
        let probability = settle(numerator / total, mines_total == 0, false)?;
        for &cell in interior {
            probabilities[cell] = probability;
        }
    }

    debug_assert!(
        {
            // However the weights are scaled, the estimates must account for
            // exactly the mines that are out there. This one line catches a
            // mis-scaled table, a wrong interior window and an off-by-one in the
            // prefix/suffix chain alike.
            let sum: f64 = probabilities.iter().sum();
            (sum - mines_total as f64).abs() < 1e-6 * (mines_total as f64).max(1.0)
        },
        "probabilities sum to {} but {} mines remain",
        probabilities.iter().sum::<f64>(),
        mines_total
    );

    Some(probabilities)
}

/// Turn a computed ratio into a reported probability without ever manufacturing
/// a certainty out of arithmetic.
///
/// `None` means the arithmetic went out of range and the caller should discard
/// the whole answer rather than show it. Otherwise a value that is certain is
/// reported exactly, and a value that is merely very small or very large is kept
/// strictly inside the open interval — because 0.0 and 1.0 are read as proof.
fn settle(ratio: f64, proven_safe: bool, proven_mine: bool) -> Option<f64> {
    if !ratio.is_finite() {
        return None;
    }
    if proven_safe {
        return Some(0.0);
    }
    if proven_mine {
        return Some(1.0);
    }
    Some(ratio.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON))
}

/// `out[j]` = the convolution of the first `j` distributions; `out[0]` is the
/// identity, so the result has one more entry than the input.
fn running_convolution<'a>(
    distributions: impl Iterator<Item = &'a Vec<f64>>,
) -> Vec<Vec<f64>> {
    let mut running = vec![vec![1.0f64]];
    for distribution in distributions {
        let next = convolve(running.last().unwrap(), distribution);
        running.push(next);
    }
    running
}

/// Distribution of the sum of two independent mine counts.
fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0f64; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        if x == 0.0 {
            continue;
        }
        for (j, &y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

/// Weights `C(n, k)` for every k in `0..=n`, rescaled to stay inside `f64`.
///
/// The true values do not fit — `C(2400, 1200)` is about 10^722 — and only ratios
/// between them are ever used, so the table can be divided by any constant. This
/// one divides by the largest weight the caller can actually reach.
///
/// Which weight that is matters. Dividing by the largest entry *overall* looks
/// natural and is wrong: on a sparse board the reachable `k` sits far out in the
/// tail, `exp(566 - 1663)` underflows to exactly zero, every layout weighs
/// nothing, and a grid of 0% reads as "all safe".
pub(crate) fn scaled_binomials(n: usize, k_min: usize, k_max: usize) -> Vec<f64> {
    let mut ln_factorial = vec![0.0f64; n + 1];
    for i in 1..=n {
        ln_factorial[i] = ln_factorial[i - 1] + (i as f64).ln();
    }
    let ln_weights: Vec<f64> = (0..=n)
        .map(|k| ln_factorial[n] - ln_factorial[k] - ln_factorial[n - k])
        .collect();

    let peak = ln_weights
        .iter()
        .take(k_max.min(n) + 1)
        .skip(k_min.min(n))
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let peak = if peak.is_finite() { peak } else { 0.0 };

    ln_weights.iter().map(|&w| (w - peak).exp()).collect()
}
