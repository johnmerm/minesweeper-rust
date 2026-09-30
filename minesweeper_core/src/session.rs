//! Everything slow, on one thread that outlives the moves.
//!
//! The exact search and the network both take long enough to be felt, and a
//! front-end's own thread has better things to do — Qt draws on it, the web
//! server answers requests on it. So they live here, behind a channel: the
//! front-end sends a [`Job`] after every move and hears back through [`Reply`]s.
//! The Qt GUI and the web server both drive it; `wasm32-unknown-unknown` has no
//! threads, which is why the wasm crate does the same work inline.
//!
//! One long-lived thread rather than one per move, for the same reason the wasm
//! crate keeps one `ConstraintSearch` in its `AppState`: the search caches
//! component solutions between boards, and a fresh instance per move starts with
//! that cache empty. It also owns the network, which corrects itself as it
//! scores and so has to be the same network from one move to the next.

use crate::probability::{certain_cells, ConstraintSearch, PatchCnn, SimUpdate};
use crate::{CellState, GameState, Minesweeper};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

/// The weights `wasm/` ships, compiled in the same way: one binary, nothing to
/// find at run time, and no second copy that can drift from the page's.
pub const WEIGHTS: &[u8] = include_bytes!("../../neural/onnx/model.bin");

/// What a cell the network has not reached yet reads.
///
/// Not zero, for the reason `wasm/src/lib.rs` gives: zero is an answer the network
/// can really give, and everywhere else it means *proven safe*.
pub const NOT_SCORED: f32 = -1.0;

/// Cells scored between checks for a newer job.
///
/// Small enough that a click abandons a stale pass within a few milliseconds,
/// large enough that the check is not the cost.
const NEURAL_CHUNK: usize = 16;

/// How often a pass in progress reports what it has so far.
const REPORT_EVERY: Duration = Duration::from_millis(120);

/// Correction rate while the solver is on screen, as `learningRate` in
/// `docs/minesweeper.js` sets it.
pub const LEARN_RATE: f32 = 0.02;

/// What to do with the next board.
pub struct Job {
    /// Echoed back on every reply, so the caller can drop answers about a board
    /// the player has since moved on from.
    pub generation: u64,
    pub game: Minesweeper,
    /// Open every cell proven safe and flag every cell proven a mine, to a
    /// fixpoint, before reporting.
    pub proof_auto_play: bool,
    /// Score the board with the network after solving it.
    pub neural: Option<NeuralJob>,
}

pub struct NeuralJob {
    /// Correct the network against the exact answer as it scores. Only honoured
    /// when the solve finished: an unfinished one has no answer to learn from.
    pub learn: bool,
    /// Once every cell is scored, play from the guesses: open below
    /// `open_below`, flag above `flag_above`. `None` leaves the board alone.
    pub auto_play: Option<(f32, f32)>,
}

/// The exact search's verdict on one board.
#[derive(Clone, Default)]
pub struct Exact {
    /// Per-cell probability, `y * width + x`, or `None` when the search gave up.
    /// There is no third case: nothing stands in for a refusal.
    pub probs: Option<Vec<f32>>,
    pub layouts: usize,
    pub nodes: usize,
    pub memory_bytes: usize,
    pub cache_hits: u32,
    pub cache_misses: u32,
}

pub enum Reply {
    /// The exact answer for `generation`. `game` is set when proof auto-play
    /// moved the board on, and is the board `exact` describes.
    Solved { generation: u64, game: Option<Minesweeper>, exact: Exact },
    /// The network's scores so far. Cells it has not reached read [`NOT_SCORED`].
    Neural {
        generation: u64,
        probs: Vec<f32>,
        total: usize,
        remaining: usize,
        /// Mean |network - exact| over the cells corrected so far this pass.
        error: Option<f32>,
    },
    /// The network played a move from its guesses — or, with both counts zero,
    /// found nothing it was sure enough about, which ends its run.
    NeuralPlayed { generation: u64, game: Minesweeper, opened: u32, flagged: u32 },
    /// The build's weights did not load; the network is off for good.
    NoNetwork,
}

/// Serve jobs until the sending side of `jobs` is dropped.
///
/// `reply` is called on this thread; getting the answer to wherever it is
/// wanted is the caller's business.
pub fn run(jobs: Receiver<Job>, reply: impl Fn(Reply)) {
    let exact = ConstraintSearch::new();
    let mut network = match PatchCnn::from_bytes(WEIGHTS) {
        Ok(network) => Some(network),
        Err(error) => {
            eprintln!("neural network unavailable: {error}");
            reply(Reply::NoNetwork);
            None
        }
    };

    let mut next = jobs.recv().ok();
    while let Some(mut job) = next.take() {
        // Only the newest board is worth solving; anything queued behind it
        // describes a position the player has already left.
        while let Ok(newer) = jobs.try_recv() {
            job = newer;
        }

        let (game, moved, solved) = if job.proof_auto_play {
            let mut game = job.game;
            let (opened, solved) = auto_reveal(&exact, &mut game);
            (game, opened > 0, solved)
        } else {
            let solved = solve(&exact, &job.game);
            (job.game, false, solved)
        };
        reply(Reply::Solved {
            generation: job.generation,
            game: moved.then(|| game.clone()),
            exact: solved.clone(),
        });

        let (Some(neural), Some(network)) = (job.neural, network.as_mut()) else {
            next = jobs.recv().ok();
            continue;
        };
        match score(network, &game, &solved, &neural, job.generation, &jobs, &reply) {
            Scored::Interrupted(newer) => next = Some(newer),
            Scored::Closed => return,
            Scored::Complete(probs) => {
                if let Some((open_below, flag_above)) = neural.auto_play {
                    let mut game = game;
                    let (opened, flagged) = neural_auto(&mut game, &probs, open_below, flag_above);
                    reply(Reply::NeuralPlayed { generation: job.generation, game, opened, flagged });
                }
                next = jobs.recv().ok();
            }
        }
    }
}

enum Scored {
    Complete(Vec<f32>),
    /// A newer job arrived; this pass was measuring a board that no longer exists.
    Interrupted(Job),
    Closed,
}

/// Score every unopened cell, reporting as it goes, until done or superseded.
fn score(
    network: &mut PatchCnn,
    game: &Minesweeper,
    exact: &Exact,
    neural: &NeuralJob,
    generation: u64,
    jobs: &Receiver<Job>,
    reply: &impl Fn(Reply),
) -> Scored {
    let mut probs = vec![NOT_SCORED; game.width * game.height];
    let mut scorer = network.scorer(game);
    let total = scorer.remaining();
    // Before the first reveal every cell carries the same prior, which teaches
    // the network nothing and would drag it towards the mean.
    let targets = exact.probs.as_ref().filter(|_| neural.learn && game.mines_generated);

    let mut reported = Instant::now();
    loop {
        match targets {
            Some(targets) => scorer.step_training(network, NEURAL_CHUNK, &mut probs, targets, LEARN_RATE),
            None => scorer.step(network, NEURAL_CHUNK, &mut probs),
        };
        let remaining = scorer.remaining();
        if remaining == 0 || reported.elapsed() >= REPORT_EVERY {
            reported = Instant::now();
            reply(Reply::Neural {
                generation,
                probs: probs.clone(),
                total,
                remaining,
                error: scorer.mean_error(),
            });
        }
        if remaining == 0 {
            return Scored::Complete(probs);
        }
        match jobs.try_recv() {
            Ok(newer) => return Scored::Interrupted(newer),
            Err(TryRecvError::Disconnected) => return Scored::Closed,
            Err(TryRecvError::Empty) => {}
        }
    }
}

/// Solve the board exactly, or report that the search gave up.
pub fn solve(search: &ConstraintSearch, game: &Minesweeper) -> Exact {
    let (tx, rx) = std::sync::mpsc::channel();
    search.calculate_with_progress(game, tx);
    let mut exact = Exact::default();
    while let Ok(update) = rx.recv() {
        if let SimUpdate::Done { probs, valid, attempts, memory_bytes, .. } = update {
            exact.layouts = valid;
            exact.nodes = attempts;
            exact.memory_bytes = memory_bytes;
            // Zero layouts is how the search says it gave up; the grid it sends
            // alongside is all zeros, which would read as a board of proofs.
            exact.probs = (valid > 0).then(|| probs.into_iter().flatten().map(|p| p as f32).collect());
            break;
        }
    }
    (exact.cache_hits, exact.cache_misses) = search.cache_counts();
    exact
}

/// Open every cell proven safe and flag every cell proven a mine, to a fixpoint.
///
/// The same loop as `AppState::auto_reveal` in `wasm/src/lib.rs`, and for the
/// same reason: iterate on [`certain_cells`], which costs no search, and pay for
/// a full solve only once it runs dry. Re-solving after every pass is what made
/// one click take 31 seconds. Returns how many cells were opened, and the solve
/// of the board it leaves behind.
pub fn auto_reveal(search: &ConstraintSearch, game: &mut Minesweeper) -> (u32, Exact) {
    if game.state != GameState::Playing || !game.mines_generated {
        return (0, solve(search, game));
    }
    let mut revealed = 0;
    loop {
        let proven = certain_cells(game);
        flag_all(game, proven.mines.iter().copied());
        let opened = reveal_all(game, proven.safe.iter().copied());
        revealed += opened;
        if game.state != GameState::Playing {
            break;
        }
        // Only a reveal is progress: a flag tells the solver nothing it did not
        // already know, so looping on flags alone would never end.
        if opened > 0 {
            continue;
        }

        // Only a finished solve proves anything.
        let Some(probs) = solve(search, game).probs else { break };
        let cells = |keep: fn(f32) -> bool| {
            let width = game.width;
            probs
                .iter()
                .enumerate()
                .filter(move |&(_, &p)| keep(p))
                .map(move |(i, _)| (i % width, i / width))
                .collect::<Vec<_>>()
        };
        let mines = cells(|p| p > 1.0 - 1e-9);
        let safe = cells(|p| p < 1e-9);
        flag_all(game, mines.into_iter());
        let opened = reveal_all(game, safe.into_iter());
        revealed += opened;
        if opened == 0 || game.state != GameState::Playing {
            break;
        }
    }
    // Leave the answer describing the board the caller is about to draw, not
    // the one it sent.
    (revealed, solve(search, game))
}

/// Play one move from the network's guesses, as `ms_neural_auto` does.
///
/// Deliberately apart from [`auto_reveal`], which acts only on proof. This acts
/// on an estimate and will sooner or later open a mine — that is how the network
/// gets measured. `probs` must cover every unopened cell; [`score`] only returns
/// a complete pass, and an unscored cell would read as the safest on the board.
pub fn neural_auto(game: &mut Minesweeper, probs: &[f32], open_below: f32, flag_above: f32) -> (u32, u32) {
    if game.state != GameState::Playing || !game.mines_generated {
        return (0, 0);
    }
    let width = game.width;
    let hidden: Vec<(usize, usize, f32)> = (0..game.height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .filter(|&(x, y)| game.grid[y][x].state == CellState::Hidden)
        .map(|(x, y)| (x, y, probs[y * width + x]))
        .collect();
    debug_assert!(hidden.iter().all(|&(_, _, p)| p >= 0.0), "an unscored cell reached auto-play");

    // Chosen before anything is applied: revealing cascades, and a cell judged
    // on this board is not one the next board should be judged by.
    let to_flag: Vec<_> = hidden.iter().filter(|c| c.2 > flag_above).map(|c| (c.0, c.1)).collect();
    let to_open: Vec<_> = hidden.iter().filter(|c| c.2 < open_below).map(|c| (c.0, c.1)).collect();
    let flagged = flag_all(game, to_flag.into_iter());
    let opened = reveal_all(game, to_open.into_iter());
    (opened, flagged)
}

/// Flag the still-hidden cells among `cells`. `toggle_flag` on a flagged cell
/// would clear it, undoing the deduction instead of recording it.
fn flag_all(game: &mut Minesweeper, cells: impl Iterator<Item = (usize, usize)>) -> u32 {
    let mut flagged = 0;
    for (x, y) in cells {
        if game.grid[y][x].state == CellState::Hidden {
            game.toggle_flag(x, y);
            flagged += 1;
        }
    }
    flagged
}

/// Reveal the still-hidden cells among `cells`.
fn reveal_all(game: &mut Minesweeper, cells: impl Iterator<Item = (usize, usize)>) -> u32 {
    let mut opened = 0;
    for (x, y) in cells {
        if game.grid[y][x].state == CellState::Hidden {
            game.reveal(x, y);
            opened += 1;
        }
    }
    opened
}

/// Largest board side a front-end offers, as `MAX_DIM` in the wasm crate. A
/// guard on how many cells a UI has to lay out, not an engine limit.
pub const MAX_DIM: usize = 200;

/// Above this many cells the network only scores once the player has picked a
/// [`Show`] mode: a pass restarts after every move, and on a huge board that keeps
/// a core busy between clicks nobody asked for. Choosing a mode is the asking.
pub const NEURAL_AUTO_MAX_CELLS: usize = 4096;

/// The network's auto-play thresholds, as on the page: open under 5%, flag over 95%.
pub const NEURAL_OPEN_BELOW: f32 = 0.05;
pub const NEURAL_FLAG_ABOVE: f32 = 0.95;

/// Which estimate is on screen — the page's `Show` select.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Show {
    #[default]
    Both,
    Exact,
    Neural,
}

impl Show {
    /// The select's option index: 0 both, 1 exact, 2 neural. Anything else is `Both`.
    pub fn from_index(index: i32) -> Self {
        match index {
            1 => Self::Exact,
            2 => Self::Neural,
            _ => Self::Both,
        }
    }

    pub fn index(self) -> i32 {
        self as i32
    }
}

/// What the network has said about the current board.
#[derive(Clone, Debug, Default)]
pub struct NeuralState {
    /// Per cell, `y * width + x`; [`NOT_SCORED`] where it has not looked yet.
    pub probs: Vec<f32>,
    pub total: usize,
    pub remaining: usize,
    /// Mean |network - exact| over the cells corrected so far this pass.
    pub error: Option<f32>,
    /// Its last auto-play pass found nothing past the thresholds.
    pub stuck: bool,
    /// What it has played this game.
    pub opened: u32,
    pub flagged: u32,
    /// The build's weights did not load; the network is off for good.
    pub broken: bool,
}

/// One game driven through a [`run`] thread: the board, the settings that decide
/// what gets asked of the session, and the latest answers about the board.
///
/// Every front-end that has threads needs exactly this bookkeeping — which
/// answers are stale, when the network is corrected and when it plays, when the
/// clock runs — and two copies of it are how front-ends drift apart. So it lives
/// here, holding data only; how any of it is worded or drawn stays with the
/// front-end.
pub struct Controller {
    pub game: Minesweeper,
    /// Bumped on every change to the board; replies about an older one are dropped.
    generation: u64,
    /// The exact answer for the current board, once it has arrived.
    pub exact: Option<Exact>,
    pub neural: NeuralState,
    pub show: Show,
    /// Whether the player has picked a mode, which lifts the size guard.
    pub show_chosen: bool,
    pub auto_play: bool,
    started: Option<Instant>,
    /// Frozen when the game ends.
    finished: Option<Duration>,
    jobs: std::sync::mpsc::Sender<Job>,
}

impl Controller {
    /// A 10x10 game with 10 mines, already submitted to `jobs`.
    pub fn new(jobs: std::sync::mpsc::Sender<Job>) -> Self {
        let mut controller = Self {
            game: Minesweeper::new(10, 10, 10),
            generation: 0,
            exact: None,
            neural: NeuralState::default(),
            show: Show::Both,
            show_chosen: false,
            auto_play: false,
            started: None,
            finished: None,
            jobs,
        };
        controller.submit();
        controller
    }

    /// Start over, clamping the size as the page does. Returns the board made.
    pub fn new_game(&mut self, width: usize, height: usize, mines: usize) -> (usize, usize, usize) {
        let w = width.clamp(3, MAX_DIM);
        let h = height.clamp(3, MAX_DIM);
        let m = mines.clamp(1, w * h - 1);
        self.game = Minesweeper::new(w, h, m);
        self.started = None;
        self.finished = None;
        let broken = self.neural.broken;
        self.neural = NeuralState { broken, ..NeuralState::default() };
        // A fresh board still has a probability — mines over cells, everywhere —
        // so it is solved like any other rather than drawn as 0%.
        self.submit();
        (w, h, m)
    }

    /// Reveal a cell the player chose. Refuses a flagged cell — never blow one up
    /// — and does nothing once the game is over. Returns whether the board changed.
    pub fn reveal(&mut self, x: usize, y: usize) -> bool {
        if !self.playable(x, y) || self.game.grid[y][x].state == CellState::Flagged {
            return false;
        }
        self.game.reveal(x, y);
        self.started.get_or_insert_with(Instant::now);
        self.submit();
        true
    }

    /// Toggle a flag. Returns whether the board changed.
    pub fn flag(&mut self, x: usize, y: usize) -> bool {
        if !self.playable(x, y) {
            return false;
        }
        self.game.toggle_flag(x, y);
        self.submit();
        true
    }

    fn playable(&self, x: usize, y: usize) -> bool {
        self.game.state == GameState::Playing && x < self.game.width && y < self.game.height
    }

    pub fn set_show(&mut self, show: Show) {
        self.show = show;
        self.show_chosen = true;
        // Whatever the last correction measured describes a mode just left.
        self.neural.error = None;
        self.submit();
    }

    pub fn set_auto_play(&mut self, on: bool) {
        self.auto_play = on;
        if on {
            self.submit();
        }
    }

    /// Whether the network should be scoring this board.
    pub fn neural_wanted(&self) -> bool {
        !self.neural.broken
            && self.show != Show::Exact
            && (self.show_chosen || self.game.width * self.game.height <= NEURAL_AUTO_MAX_CELLS)
    }

    /// Whether an answer is still on its way.
    pub fn busy(&self) -> bool {
        self.exact.is_none() || (self.neural_wanted() && self.neural.remaining > 0)
    }

    /// Time on the clock: from the player's first reveal until the game ended.
    pub fn elapsed(&self) -> Duration {
        self.finished
            .or_else(|| self.started.map(|s| s.elapsed()))
            .unwrap_or_default()
    }

    /// Whether the clock is running.
    pub fn timing(&self) -> bool {
        self.started.is_some() && self.finished.is_none()
    }

    fn note_end(&mut self) {
        if self.game.state != GameState::Playing && self.finished.is_none() {
            self.finished = Some(self.started.map(|s| s.elapsed()).unwrap_or_default());
        }
    }

    /// The board changed: forget every answer about the old one and ask again.
    pub fn submit(&mut self) {
        self.note_end();
        self.generation += 1;
        self.exact = None;
        let over = self.game.state != GameState::Playing;
        // A finished game is not rescored: the last pass is what the network
        // thought when it mattered, and that is what stays on screen.
        if !over {
            self.neural.probs = vec![NOT_SCORED; self.game.width * self.game.height];
            self.neural.total = 0;
            self.neural.remaining = 0;
            self.neural.stuck = false;
        }
        let neural = (self.neural_wanted() && !over).then(|| NeuralJob {
            // Corrected only while the solver is on screen: in `Neural` the point
            // is to watch the network unaided.
            learn: self.show != Show::Neural,
            auto_play: (self.auto_play && self.show == Show::Neural)
                .then_some((NEURAL_OPEN_BELOW, NEURAL_FLAG_ABOVE)),
        });
        let _ = self.jobs.send(Job {
            generation: self.generation,
            game: self.game.clone(),
            // Proof-driven auto-play only when proof is what is on screen.
            proof_auto_play: self.auto_play && self.show != Show::Neural,
            neural,
        });
    }

    /// Take in an answer from the session. Returns whether anything changed.
    pub fn on_reply(&mut self, reply: Reply) -> bool {
        match reply {
            Reply::Solved { generation, game, exact } if generation == self.generation => {
                if let Some(game) = game {
                    // Auto-play moved the board on; this is the board `exact` is about.
                    self.game = game;
                    self.note_end();
                }
                self.exact = Some(exact);
            }
            Reply::Neural { generation, probs, total, remaining, error } if generation == self.generation => {
                self.neural.probs = probs;
                self.neural.total = total;
                self.neural.remaining = remaining;
                self.neural.error = error;
            }
            Reply::NeuralPlayed { generation, opened: 0, flagged: 0, .. } if generation == self.generation => {
                self.neural.stuck = true;
            }
            Reply::NeuralPlayed { generation, game, opened, flagged } if generation == self.generation => {
                self.neural.opened += opened;
                self.neural.flagged += flagged;
                self.game = game;
                // Rescoring the new board is what drives the next move.
                self.submit();
            }
            Reply::NoNetwork => self.neural.broken = true,
            _ => return false, // about a board that no longer exists
        }
        true
    }
}
