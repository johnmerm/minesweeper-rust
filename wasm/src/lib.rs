//! WebAssembly front-end bindings for [`minesweeper_core`].
//!
//! This crate is deliberately built **without** `wasm-bindgen`: it exports a
//! small, hand-rolled C ABI so the resulting `.wasm` can be loaded by a plain
//! `WebAssembly.instantiate(...)` call from a single static HTML page. No
//! bundler, no generated JS glue, no server — the page and the module can be
//! served straight from a CDN such as `raw.githack.com`.
//!
//! # ABI overview
//!
//! All game state lives in this module; JavaScript only sends commands and
//! reads back two flat buffers plus a small stats array.
//!
//! | Export | Purpose |
//! |--------|---------|
//! | `ms_seed(hi, lo)` | Seed the PRNG (call once before the first reveal) |
//! | `ms_new(w, h, mines)` | Start a new game |
//! | `ms_reveal(x, y)` / `ms_flag(x, y)` | Player actions |
//! | `ms_compute(mode)` | Recompute mine probabilities |
//! | `ms_auto_reveal(mode)` | Reveal every provably-safe cell, repeatedly |
//! | `ms_cells_ptr()` | `u8[width * height]` — see [`encode_cell`] |
//! | `ms_probs_ptr()` | `f32[width * height]` — mine probability per cell |
//! | `ms_stats_ptr()` | `u32[9]` — see the `STAT_*` constants |
//! | `ms_width()` / `ms_height()` / `ms_mines()` / `ms_state()` | Scalars |
//!
//! Buffers are reallocated by `ms_new`, and the module's linear memory can be
//! grown by any call, so JavaScript must re-read the pointers (and re-create its
//! typed-array views) after every call into the module.

use std::cell::RefCell;
use std::sync::mpsc::Sender;

use minesweeper_core::probability::{certain_cells, BoardScorer, ConstraintSearch, MonteCarlo, PatchCnn, SimUpdate};
use minesweeper_core::{CellContent, CellState, GameState, Minesweeper};

mod rng;

/// Probability strategy requested by the caller of [`ms_compute`].
mod mode {
    /// Exact constraint search, falling back to Monte Carlo if it finds no layout.
    pub const AUTO: u32 = 0;
    /// Monte Carlo sampling only.
    pub const MONTE_CARLO: u32 = 1;
    /// Constraint search only.
    pub const CONSTRAINT_SEARCH: u32 = 2;
}

// Indices into the `u32` array exposed by [`ms_stats_ptr`].
const STAT_MC_VALID: usize = 0;
const STAT_MC_ATTEMPTS: usize = 1;
const STAT_MC_MEMORY: usize = 2;
const STAT_CS_VALID: usize = 3;
const STAT_CS_ATTEMPTS: usize = 4;
const STAT_CS_MEMORY: usize = 5;
/// Which strategy's numbers ended up in the probability buffer: one of `mode::*`
/// (never `AUTO` — it is resolved to the strategy actually used).
const STAT_USED: usize = 6;
/// Component solves served from the cache since the page loaded, and solves that
/// had to be done. Cumulative, not per move.
const STAT_CACHE_HITS: usize = 7;
const STAT_CACHE_MISSES: usize = 8;
const STAT_LEN: usize = 9;

/// What a cell the network has not reached yet reads in `ms_neural_probs_ptr`.
///
/// Deliberately not zero. Zero is a probability the network can genuinely
/// return, and in every other buffer here it means *proven safe* — so a buffer
/// that started at zero would show a whole board the network had never looked at
/// as one it had declared harmless.
const NOT_SCORED: f32 = -1.0;

/// Cell encodings written into the buffer returned by [`ms_cells_ptr`].
mod cell_code {
    /// `0..=8` are visible cells showing that neighbour-mine count.
    pub const VISIBLE_MINE: u8 = 9;
    pub const HIDDEN: u8 = 10;
    pub const FLAGGED: u8 = 11;
}

/// Largest board dimension we accept.
///
/// Not a limit of the engine — it is a guard against a typo asking for a board
/// with a million cells, each of which becomes a DOM node. 200x200 is 40 000
/// cells, which is already slow to lay out; the estimators cope, since they are
/// bounded and mostly care about the border rather than the area.
const MAX_DIM: usize = 200;

struct AppState {
    game: Minesweeper,
    /// The neural estimator, once JavaScript has handed over its weights.
    network: Option<PatchCnn>,
    /// Scratch space JavaScript writes the weights into.
    incoming: Vec<u8>,
    /// The network's guess per cell, alongside `probs` from the exact search.
    neural_probs: Vec<f32>,
    /// The scoring pass in flight, if any.
    scoring: Option<BoardScorer>,
    cells: Vec<u8>,
    probs: Vec<f32>,
    stats: [u32; STAT_LEN],
    /// Kept across moves, not rebuilt per call: its component cache is what makes
    /// a second look at the same board nearly free, and a fresh instance would
    /// start empty every time.
    exact: ConstraintSearch,
}

impl AppState {
    fn new(width: usize, height: usize, mines: usize) -> Self {
        let mut state = Self {
            game: Minesweeper::new(width, height, mines),
            cells: vec![cell_code::HIDDEN; width * height],
            probs: vec![0.0; width * height],
            stats: [0; STAT_LEN],
            exact: ConstraintSearch::new(),
            network: None,
            incoming: Vec::new(),
            neural_probs: vec![NOT_SCORED; width * height],
            scoring: None,
        };
        state.sync_cells();
        state
    }

    /// Re-encode the board into the flat byte buffer JavaScript reads.
    fn sync_cells(&mut self) {
        self.cells.clear();
        for row in &self.game.grid {
            self.cells.extend(row.iter().map(encode_cell));
        }
    }

    /// Recompute mine probabilities with the requested strategy.
    fn compute(&mut self, mode: u32) {
        self.stats = [0; STAT_LEN];

        // Exact enumeration runs first: it is usually far cheaper than a million
        // Monte Carlo draws and its answers are exact, so in AUTO mode we only
        // pay for sampling when the search comes back with no valid layout
        // (which happens on a fresh board, where there are no constraints yet).
        let cs = (mode != mode::MONTE_CARLO)
            .then(|| run_sync(|tx| self.exact.calculate_with_progress(&self.game, tx)));
        let cs_ok = cs.as_ref().map_or(false, |run| run.valid > 0);
        let mc = (mode == mode::MONTE_CARLO || (mode == mode::AUTO && !cs_ok))
            .then(|| run_sync(|tx| MonteCarlo::new().calculate_with_progress(&self.game, tx)));

        if let Some(run) = &mc {
            self.stats[STAT_MC_VALID] = run.valid as u32;
            self.stats[STAT_MC_ATTEMPTS] = run.attempts as u32;
            self.stats[STAT_MC_MEMORY] = run.memory_bytes as u32;
        }
        if let Some(run) = &cs {
            self.stats[STAT_CS_VALID] = run.valid as u32;
            self.stats[STAT_CS_ATTEMPTS] = run.attempts as u32;
            self.stats[STAT_CS_MEMORY] = run.memory_bytes as u32;
        }

        let (probs, used) = match (cs, mc) {
            (Some(cs), _) if cs.valid > 0 => (cs.probs, mode::CONSTRAINT_SEARCH),
            (_, Some(mc)) => (mc.probs, mode::MONTE_CARLO),
            (Some(cs), None) => (cs.probs, mode::CONSTRAINT_SEARCH),
            (None, None) => (Vec::new(), mode::AUTO),
        };
        self.stats[STAT_USED] = used;
        let (hits, misses) = self.exact.cache_counts();
        self.stats[STAT_CACHE_HITS] = hits;
        self.stats[STAT_CACHE_MISSES] = misses;
        self.store_probs(&probs);
    }

    fn store_probs(&mut self, probs: &[Vec<f64>]) {
        self.probs.clear();
        for y in 0..self.game.height {
            for x in 0..self.game.width {
                let p = probs.get(y).and_then(|row| row.get(x)).copied().unwrap_or(0.0);
                self.probs.push(p as f32);
            }
        }
    }

    /// Reveal every cell that can be proven safe, repeatedly, and return how
    /// many were revealed.
    ///
    /// Each reveal changes the board, so this has to iterate — but re-running the
    /// full estimator on every iteration is what made this hang: on a 30x30 board
    /// with 250 mines one click took 47 passes at ~666 ms each, over 31 seconds,
    /// because a half-open cascade front is the exact search's worst case.
    ///
    /// Instead, iterate on [`certain_cells`], which proves safety with local
    /// rules only and costs no search, and pay for a full solve only once
    /// propagation has run dry — that last step is what catches the cells only a
    /// full enumeration can prove safe. In practice that turns dozens of solves
    /// into one or two.
    fn auto_reveal(&mut self, mode: u32) -> u32 {
        if self.game.state != GameState::Playing || !self.game.mines_generated {
            return 0;
        }

        let mut revealed = 0;
        loop {
            // Cheap: everything the local rules can prove, to a fixpoint. Cells
            // proven to *be* mines get flagged — the same deduction, and the
            // player would only have to make it again by hand.
            let proven = certain_cells(&self.game);
            self.flag_all(&proven.mines);
            let opened = self.reveal_all(&proven.safe);
            revealed += opened;
            if self.game.state != GameState::Playing {
                break;
            }
            // Only a reveal is progress: a flag tells the estimator nothing it
            // did not already know, so looping on flags alone would never end.
            if opened > 0 {
                continue;
            }

            // Propagation is exhausted, so it is worth one full solve to see
            // whether anything else is provably safe.
            self.sync_cells();
            self.compute(mode);

            // Only the exact search proves anything. A sampled 0% just means no
            // draw happened to put a mine there, and opening on that would
            // eventually detonate one — so if the search bailed out and we are
            // looking at a Monte Carlo estimate, stop here.
            if self.stats[STAT_USED] != mode::CONSTRAINT_SEARCH {
                break;
            }

            let width = self.game.width;
            let certain: Vec<(usize, usize)> = (0..self.game.height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .filter(|&(x, y)| self.probs[y * width + x] > 1.0 - 1e-9)
                .collect();
            self.flag_all(&certain);

            let deduced: Vec<(usize, usize)> = (0..self.game.height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .filter(|&(x, y)| self.probs[y * width + x] < 1e-9)
                .collect();
            let opened = self.reveal_all(&deduced);
            revealed += opened;
            if opened == 0 || self.game.state != GameState::Playing {
                break;
            }
        }

        self.sync_cells();
        // Leave the probability buffer describing the board JavaScript is about
        // to draw, not the one we started from.
        self.compute(mode);
        revealed
    }

    /// Flag the still-hidden cells among `cells`, returning how many were flagged.
    ///
    /// Only hidden ones: `toggle_flag` would otherwise clear a flag already
    /// standing on the cell, undoing the deduction instead of recording it.
    fn flag_all(&mut self, cells: &[(usize, usize)]) -> u32 {
        let mut flagged = 0;
        for &(x, y) in cells {
            if self.game.grid[y][x].state == CellState::Hidden {
                self.game.toggle_flag(x, y);
                flagged += 1;
            }
        }
        flagged
    }

    /// Reveal the still-hidden cells among `cells`, returning how many opened.
    fn reveal_all(&mut self, cells: &[(usize, usize)]) -> u32 {
        let mut opened = 0;
        for &(x, y) in cells {
            if self.game.grid[y][x].state == CellState::Hidden {
                self.game.reveal(x, y);
                opened += 1;
            }
        }
        opened
    }
}

/// Encode one cell as a single byte for the JavaScript side.
fn encode_cell(cell: &minesweeper_core::Cell) -> u8 {
    match cell.state {
        CellState::Hidden => cell_code::HIDDEN,
        CellState::Flagged => cell_code::FLAGGED,
        CellState::Visible => match cell.content {
            CellContent::Mine => cell_code::VISIBLE_MINE,
            CellContent::Empty(n) => n.min(8),
        },
    }
}

/// Result of driving one probability strategy to completion.
struct Run {
    probs: Vec<Vec<f64>>,
    valid: usize,
    attempts: usize,
    memory_bytes: usize,
}

/// Drive a `calculate_with_progress` call to its `Done` update.
///
/// Both strategies are fully synchronous, so the sender has already finished by
/// the time we drain the channel — no threads are involved, which matters on
/// `wasm32-unknown-unknown` where they are unavailable.
fn run_sync(run: impl FnOnce(Sender<SimUpdate>)) -> Run {
    let (tx, rx) = std::sync::mpsc::channel();
    run(tx);
    let mut result = Run { probs: Vec::new(), valid: 0, attempts: 0, memory_bytes: 0 };
    while let Ok(update) = rx.recv() {
        if let SimUpdate::Done { probs, valid, attempts, memory_bytes, .. } = update {
            result = Run { probs, valid, attempts, memory_bytes };
            break;
        }
    }
    result
}

thread_local! {
    static STATE: RefCell<AppState> = RefCell::new(AppState::new(10, 10, 10));
}

fn with_state<R>(f: impl FnOnce(&mut AppState) -> R) -> R {
    STATE.with(|state| f(&mut state.borrow_mut()))
}

/// Seed the PRNG from JavaScript (`Math.random()` / `Date.now()`).
///
/// The module has no access to `crypto.getRandomValues` without JS glue, so the
/// host supplies the entropy once and [`rng`] expands it from there.
#[no_mangle]
pub extern "C" fn ms_seed(hi: u32, lo: u32) {
    rng::seed(((hi as u64) << 32) | lo as u64);
}

/// Start a new game. Dimensions are clamped to sane values.
#[no_mangle]
pub extern "C" fn ms_new(width: u32, height: u32, mines: u32) {
    let w = (width as usize).clamp(3, MAX_DIM);
    let h = (height as usize).clamp(3, MAX_DIM);
    let m = (mines as usize).clamp(1, w * h - 1);
    with_state(|state| {
        // The weights describe the game, not this board, and fetching them again
        // costs half a megabyte. Everything else starts over.
        let network = state.network.take();
        *state = AppState::new(w, h, m);
        state.network = network;
    });
}

#[no_mangle]
pub extern "C" fn ms_reveal(x: u32, y: u32) {
    with_state(|state| {
        if (x as usize) < state.game.width && (y as usize) < state.game.height {
            state.game.reveal(x as usize, y as usize);
            state.sync_cells();
        }
    });
}

#[no_mangle]
pub extern "C" fn ms_flag(x: u32, y: u32) {
    with_state(|state| {
        if (x as usize) < state.game.width && (y as usize) < state.game.height {
            state.game.toggle_flag(x as usize, y as usize);
            state.sync_cells();
        }
    });
}

/// Recompute probabilities; `mode` is one of the `mode::*` constants.
#[no_mangle]
pub extern "C" fn ms_compute(mode: u32) {
    with_state(|state| state.compute(mode));
}

/// Reveal all provably-safe cells; returns how many were revealed.
/// `mode` selects the estimator used between passes, as in [`ms_compute`].
#[no_mangle]
pub extern "C" fn ms_auto_reveal(mode: u32) -> u32 {
    with_state(|state| state.auto_reveal(mode))
}

#[no_mangle]
pub extern "C" fn ms_width() -> u32 {
    with_state(|state| state.game.width as u32)
}

#[no_mangle]
pub extern "C" fn ms_height() -> u32 {
    with_state(|state| state.game.height as u32)
}

#[no_mangle]
pub extern "C" fn ms_mines() -> u32 {
    with_state(|state| state.game.mines_count as u32)
}

/// `0` = playing, `1` = won, `2` = lost.
#[no_mangle]
pub extern "C" fn ms_state() -> u32 {
    with_state(|state| match state.game.state {
        GameState::Playing => 0,
        GameState::Won => 1,
        GameState::Lost => 2,
    })
}

/// Number of flags currently placed, for the "mines remaining" counter.
#[no_mangle]
pub extern "C" fn ms_flags() -> u32 {
    with_state(|state| {
        state
            .game
            .grid
            .iter()
            .flatten()
            .filter(|cell| cell.state == CellState::Flagged)
            .count() as u32
    })
}

/// Pointer to `width * height` cell bytes. Re-read after every call.
#[no_mangle]
pub extern "C" fn ms_cells_ptr() -> *const u8 {
    with_state(|state| state.cells.as_ptr())
}

/// Pointer to `width * height` `f32` mine probabilities. Re-read after every call.
#[no_mangle]
pub extern "C" fn ms_probs_ptr() -> *const f32 {
    with_state(|state| state.probs.as_ptr())
}

/// Pointer to the 7-entry `u32` statistics array. Re-read after every call.
#[no_mangle]
pub extern "C" fn ms_stats_ptr() -> *const u32 {
    with_state(|state| state.stats.as_ptr())
}

#[no_mangle]
pub extern "C" fn ms_stats_len() -> u32 {
    STAT_LEN as u32
}

/// Reserve `len` bytes for the network's weights and return where to write them.
#[no_mangle]
pub extern "C" fn ms_model_buffer(len: u32) -> *mut u8 {
    with_state(|state| {
        state.incoming.clear();
        state.incoming.resize(len as usize, 0);
        state.incoming.as_mut_ptr()
    })
}

/// Parse the bytes written into that buffer. Returns 1 on success, 0 on failure.
#[no_mangle]
pub extern "C" fn ms_model_load() -> u32 {
    with_state(|state| {
        let bytes = std::mem::take(&mut state.incoming);
        match PatchCnn::from_bytes(&bytes) {
            Ok(network) => {
                state.network = Some(network);
                1
            }
            Err(_) => 0,
        }
    })
}

/// Whether a network has been loaded.
#[no_mangle]
pub extern "C" fn ms_model_ready() -> u32 {
    with_state(|state| state.network.is_some() as u32)
}

/// Pointer to `width * height` f32 network predictions. Re-read after every call.
#[no_mangle]
pub extern "C" fn ms_neural_probs_ptr() -> *const f32 {
    with_state(|state| state.neural_probs.as_ptr())
}

/// Start scoring the board. Returns the number of cells that will be scored.
///
/// Any pass already running is abandoned: it was measuring a board that no longer
/// exists, and half of one position beside half of another is not a reading of
/// anything.
#[no_mangle]
pub extern "C" fn ms_neural_begin() -> u32 {
    with_state(|state| {
        let Some(network) = &state.network else {
            state.scoring = None;
            return 0;
        };
        state.neural_probs.clear();
        state.neural_probs.resize(state.game.width * state.game.height, NOT_SCORED);
        let scorer = network.scorer(&state.game);
        let total = scorer.remaining() as u32;
        state.scoring = Some(scorer);
        total
    })
}

/// Score up to `budget` more cells. Returns how many cells are still to do.
///
/// The caller decides the budget — enough to make progress, few enough to give
/// the frame back. Cells not yet scored read -1 in the buffer, so the display can
/// tell "not known yet" from "nearly zero".
#[no_mangle]
pub extern "C" fn ms_neural_step(budget: u32) -> u32 {
    with_state(|state| {
        let (Some(network), Some(scorer)) = (&state.network, &mut state.scoring) else {
            return 0;
        };
        scorer.step(network, budget as usize, &mut state.neural_probs);
        let left = scorer.remaining() as u32;
        if left == 0 {
            state.scoring = None;
        }
        left
    })
}

/// Teach the network from the exact probabilities currently in `ms_probs_ptr`.
///
/// Every position the solver scores is a perfectly labelled example that cost
/// nothing extra to produce, so the network can be corrected as the game goes on.
/// Only the output layer moves — see `PatchCnn::learn`. Returns the mean error
/// before the step, scaled by 10000 so it can come back as an integer.
#[no_mangle]
pub extern "C" fn ms_neural_learn(rate_millis: u32) -> u32 {
    with_state(|state| {
        let Some(network) = &mut state.network else { return 0 };
        if !state.game.mines_generated {
            return 0;
        }
        let width = state.game.width;
        let exact: Vec<Vec<f64>> = (0..state.game.height)
            .map(|y| (0..width).map(|x| state.probs[y * width + x] as f64).collect())
            .collect();
        let rate = rate_millis as f32 / 1000.0;
        match network.learn_from_board(&state.game, &exact, rate) {
            Some(error) => (error * 10_000.0) as u32,
            None => 0,
        }
    })
}
