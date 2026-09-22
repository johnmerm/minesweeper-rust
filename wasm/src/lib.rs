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
//! | `ms_stats_ptr()` | `u32[7]` — see the `STAT_*` constants |
//! | `ms_width()` / `ms_height()` / `ms_mines()` / `ms_state()` | Scalars |
//!
//! Buffers are reallocated by `ms_new`, and the module's linear memory can be
//! grown by any call, so JavaScript must re-read the pointers (and re-create its
//! typed-array views) after every call into the module.

use std::cell::RefCell;
use std::sync::mpsc::Sender;

use minesweeper_core::probability::{certain_cells, ConstraintSearch, MonteCarlo, SimUpdate};
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
const STAT_LEN: usize = 7;

/// Cell encodings written into the buffer returned by [`ms_cells_ptr`].
mod cell_code {
    /// `0..=8` are visible cells showing that neighbour-mine count.
    pub const VISIBLE_MINE: u8 = 9;
    pub const HIDDEN: u8 = 10;
    pub const FLAGGED: u8 = 11;
}

/// Largest board we accept, mirroring the clamp used by the Actix front-end.
const MAX_DIM: usize = 50;

struct AppState {
    game: Minesweeper,
    cells: Vec<u8>,
    probs: Vec<f32>,
    stats: [u32; STAT_LEN],
}

impl AppState {
    fn new(width: usize, height: usize, mines: usize) -> Self {
        let mut state = Self {
            game: Minesweeper::new(width, height, mines),
            cells: vec![cell_code::HIDDEN; width * height],
            probs: vec![0.0; width * height],
            stats: [0; STAT_LEN],
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
            .then(|| run_sync(|tx| ConstraintSearch::new().calculate_with_progress(&self.game, tx)));
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
            // Cheap: everything the local rules can prove, to a fixpoint.
            let safe = certain_cells(&self.game).safe;
            let opened = self.reveal_all(&safe);
            revealed += opened;
            if self.game.state != GameState::Playing {
                break;
            }
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
    with_state(|state| *state = AppState::new(w, h, m));
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
