//! Desktop front-end: the same game and the same two estimators as the page in
//! `docs/`, drawn by Qt.
//!
//! The slow work — the exact search and the network — runs on the thread in
//! [`minesweeper_core::session`], which posts its answers back here through `queued_callback`. This
//! side keeps the board, decides what each cell looks like, and repaints only
//! the cells whose appearance changed: on a 200x200 board repainting all 40 000
//! every move is what makes a front-end slow, not the estimators.

use cstr::cstr;
use minesweeper_core::{CellContent, CellState, GameState, Minesweeper};
use qmetaobject::prelude::*;
use qmetaobject::{queued_callback, QPointer, SimpleListItem, SimpleListModel};
use std::cell::RefCell;
use minesweeper_core::session::{self, Controller, Exact, Reply, Show};

/// A cell on a board the solver could not finish. Deliberately not the grey that
/// `prob_color(0.0)` produces, which is what "certainly safe" looks like.
const UNKNOWN: &str = "#c6cbd6";

/// One cell, as QML draws it.
#[derive(Default, Clone, PartialEq)]
struct CellView {
    text: String,
    fg: String,
    bg: String,
    /// Unopened: drawn raised, and clickable.
    raised: bool,
    /// Unopened and next to a number: the cells the solver actually reasons about.
    border: bool,
    /// The exact probability, `?` when unsolved, empty when not shown.
    prob: String,
    /// The network's guess, empty when not shown.
    guess: String,
}

impl SimpleListItem for CellView {
    fn get(&self, role: i32) -> QVariant {
        match role {
            0 => QString::from(self.text.as_str()).into(),
            1 => QString::from(self.fg.as_str()).into(),
            2 => QString::from(self.bg.as_str()).into(),
            3 => self.raised.into(),
            4 => self.border.into(),
            5 => QString::from(self.prob.as_str()).into(),
            6 => QString::from(self.guess.as_str()).into(),
            _ => QVariant::default(),
        }
    }

    fn names() -> Vec<QByteArray> {
        ["text", "fg", "bg", "raised", "border", "prob", "guess"]
            .iter()
            .map(|&name| QByteArray::from(name))
            .collect()
    }
}

#[derive(QObject, Default)]
struct MinesweeperGui {
    base: qt_base_class!(trait QObject),

    board_width: qt_property!(i32; NOTIFY board_changed),
    board_height: qt_property!(i32; NOTIFY board_changed),
    cells: qt_property!(RefCell<SimpleListModel<CellView>>; CONST),

    /// `Playing` / `You won!` / `Boom — game over`.
    status_text: qt_property!(QString; NOTIFY status_changed),
    /// 0 playing, 1 won, 2 lost — QML picks the colour.
    status_kind: qt_property!(i32; NOTIFY status_changed),
    mines_left: qt_property!(i32; NOTIFY status_changed),
    timer_text: qt_property!(QString; NOTIFY status_changed),
    timer_running: qt_property!(bool; NOTIFY status_changed),
    /// What the exact search did, or that it is still working.
    sim_text: qt_property!(QString; NOTIFY status_changed),
    /// 0 solved, 1 refused, 2 calculating.
    sim_kind: qt_property!(i32; NOTIFY status_changed),
    neural_note: qt_property!(QString; NOTIFY status_changed),

    show_probs: qt_property!(bool; NOTIFY settings_changed WRITE set_show_probs),
    auto_play: qt_property!(bool; NOTIFY settings_changed WRITE set_auto_play),
    flag_mode: qt_property!(bool; NOTIFY settings_changed),
    /// Index into Both / Exact / Neural.
    show_mode: qt_property!(i32; NOTIFY settings_changed WRITE set_show_mode),
    /// Whether each label is on screen, so QML can place the guess in the
    /// corner the exact value would otherwise take.
    show_exact: qt_property!(bool; NOTIFY settings_changed),
    show_neural: qt_property!(bool; NOTIFY settings_changed),

    board_changed: qt_signal!(),
    status_changed: qt_signal!(),
    settings_changed: qt_signal!(),

    init: qt_method!(fn(&mut self)),
    reveal: qt_method!(fn(&mut self, index: i32)),
    flag: qt_method!(fn(&mut self, index: i32)),
    reset: qt_method!(fn(&mut self, w: i32, h: i32, m: i32)),
    hover_text: qt_method!(fn(&self, index: i32) -> QString),
    tick: qt_method!(fn(&mut self)),

    /// The game, and the session thread answering questions about it.
    ctl: Option<Controller>,
    /// What each cell currently shows, so an unchanged cell is not pushed to QML.
    painted: Vec<CellView>,
}

impl MinesweeperGui {
    fn init(&mut self) {
        self.show_probs = true;
        let this = QPointer::from(&*self);
        let deliver = queued_callback(move |reply: Reply| {
            if let Some(this) = this.as_pinned() {
                this.borrow_mut().on_reply(reply);
            }
        });
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || session::run(rx, deliver));
        self.ctl = Some(Controller::new(tx));
        self.reset(10, 10, 10);
    }

    fn reset(&mut self, w: i32, h: i32, m: i32) {
        let Some(ctl) = &mut self.ctl else { return };
        let (w, h, _) = ctl.new_game(w.max(0) as usize, h.max(0) as usize, m.max(0) as usize);
        self.board_width = w as i32;
        self.board_height = h as i32;
        // A reset rather than per-cell changes: the delegates are for a board of
        // a different shape.
        self.painted = vec![CellView::default(); w * h];
        self.cells.borrow_mut().reset_data(self.painted.clone());
        self.board_changed();
        self.repaint();
    }

    fn cell_at(&self, index: i32) -> Option<(usize, usize)> {
        let game = &self.ctl.as_ref()?.game;
        let index = usize::try_from(index).ok()?;
        (index < game.width * game.height).then(|| (index % game.width, index / game.width))
    }

    fn reveal(&mut self, index: i32) {
        if self.flag_mode {
            return self.flag(index);
        }
        let Some((x, y)) = self.cell_at(index) else { return };
        if self.ctl.as_mut().is_some_and(|ctl| ctl.reveal(x, y)) {
            self.repaint();
        }
    }

    fn flag(&mut self, index: i32) {
        let Some((x, y)) = self.cell_at(index) else { return };
        if self.ctl.as_mut().is_some_and(|ctl| ctl.flag(x, y)) {
            self.repaint();
        }
    }

    fn on_reply(&mut self, reply: Reply) {
        if self.ctl.as_mut().is_some_and(|ctl| ctl.on_reply(reply)) {
            self.repaint();
        }
    }

    fn set_show_probs(&mut self, on: bool) {
        self.show_probs = on;
        self.repaint();
    }

    fn set_auto_play(&mut self, on: bool) {
        self.auto_play = on;
        if let Some(ctl) = &mut self.ctl {
            ctl.set_auto_play(on);
        }
        self.repaint();
    }

    fn set_show_mode(&mut self, index: i32) {
        self.show_mode = index;
        if let Some(ctl) = &mut self.ctl {
            ctl.set_show(Show::from_index(index));
        }
        self.repaint();
    }

    fn tick(&mut self) {
        let Some(ctl) = &self.ctl else { return };
        let secs = ctl.elapsed().as_secs();
        self.timer_text = format!("{}:{:02}", secs / 60, secs % 60).into();
        self.timer_running = ctl.timing();
        self.status_changed();
    }

    /// Recompute every cell's appearance and push the ones that changed.
    fn repaint(&mut self) {
        let Some(ctl) = &self.ctl else { return };
        let game = &ctl.game;
        let exact_on = ctl.show != Show::Neural;
        let neural_on = ctl.neural_wanted();
        let over = game.state != GameState::Playing;
        let solved = ctl.exact.as_ref().and_then(|e| e.probs.as_ref());
        let calculating = ctl.exact.is_none();

        {
            let mut model = self.cells.borrow_mut();
            for y in 0..game.height {
                for x in 0..game.width {
                    let i = y * game.width + x;
                    let exact = solved.map(|p| p[i]);
                    let guess = ctl.neural.probs.get(i).copied().filter(|&g| neural_on && g >= 0.0);
                    let view = cell_view(game, x, y, exact_on, exact, calculating, guess, over, self.show_probs);
                    if self.painted[i] != view {
                        self.painted[i] = view.clone();
                        model.change_line(i, view);
                    }
                }
            }
        }

        let flags = game.grid.iter().flatten().filter(|c| c.state == CellState::Flagged).count();
        self.mines_left = game.mines_count as i32 - flags as i32;
        (self.status_text, self.status_kind) = match game.state {
            GameState::Playing => ("Playing".into(), 0),
            GameState::Won => ("You won!".into(), 1),
            GameState::Lost => ("Boom — game over".into(), 2),
        };
        (self.sim_text, self.sim_kind) = describe_exact(ctl.exact.as_ref());
        self.neural_note = describe_neural(ctl).into();
        self.show_exact = exact_on;
        self.show_neural = neural_on;
        self.settings_changed();
        self.tick();
    }

    /// The line under the board while the pointer is over a cell.
    fn hover_text(&self, index: i32) -> QString {
        let (Some(ctl), Some((x, y))) = (&self.ctl, self.cell_at(index)) else {
            return QString::default();
        };
        if ctl.game.grid[y][x].state == CellState::Visible {
            return QString::default();
        }
        let i = y * ctl.game.width + x;
        let mut parts = Vec::new();
        if ctl.show != Show::Neural {
            parts.push(match &ctl.exact {
                None => "Mine probability: calculating…".to_string(),
                Some(Exact { probs: Some(p), .. }) => format!("Mine probability: {:.1}%", p[i] * 100.0),
                Some(_) => "Mine probability: not known — the search did not finish".to_string(),
            });
        }
        if ctl.neural_wanted() {
            let guess = ctl.neural.probs.get(i).copied().unwrap_or(-1.0);
            parts.push(if guess >= 0.0 {
                format!("network: {:.1}%", guess * 100.0)
            } else {
                "network: …".to_string()
            });
        }
        parts.join(" · ").into()
    }
}

/// The solver line, and whether it reports a solve (0), a refusal (1) or a wait (2).
fn describe_exact(exact: Option<&Exact>) -> (QString, i32) {
    let Some(exact) = exact else { return ("calculating…".into(), 2) };
    if exact.probs.is_none() {
        let text = format!(
            "no exact answer within {} nodes — showing none rather than guessing",
            thousands(exact.nodes)
        );
        return (text.into(), 1);
    }
    let looked = exact.cache_hits + exact.cache_misses;
    let reuse = if looked > 0 {
        format!(" · {}% reused", (100 * exact.cache_hits + looked / 2) / looked)
    } else {
        String::new()
    };
    let text = format!(
        "exact: {} region layouts / {} nodes [{}]{reuse}",
        thousands(exact.layouts),
        thousands(exact.nodes),
        fmt_memory(exact.memory_bytes)
    );
    (text.into(), 0)
}

/// The network line, worded as `describeNeural` in `docs/minesweeper.js`.
fn describe_neural(ctl: &Controller) -> String {
    let neural = &ctl.neural;
    if neural.broken {
        return "network: this build carries no usable weights".into();
    }
    if ctl.show != Show::Exact && !ctl.neural_wanted() {
        return "network: not scored automatically on a board this large — pick a mode to ask for it".into();
    }
    if !ctl.neural_wanted() {
        return String::new();
    }
    if neural.total == 0 {
        return "network: …".into();
    }
    let done = neural.total - neural.remaining;
    let mut text = if neural.remaining > 0 {
        format!("network: {} / {} cells…", thousands(done), thousands(neural.total))
    } else {
        format!("network: {} cells", thousands(neural.total))
    };
    if let Some(error) = neural.error {
        text += &format!(" · off by {:.1} points, corrected", error * 100.0);
    } else if ctl.show == Show::Neural {
        text += " · uncorrected";
    }
    if neural.opened + neural.flagged > 0 {
        text += &format!(" · it has opened {} and flagged {}", neural.opened, neural.flagged);
        match ctl.game.state {
            GameState::Lost => text += ", then hit a mine",
            GameState::Won => text += ", and won",
            GameState::Playing => {}
        }
    }
    if neural.stuck {
        text += " · nothing it is sure enough about";
    }
    text
}

/// How one cell looks, from everything known about it.
#[allow(clippy::too_many_arguments)]
fn cell_view(
    game: &Minesweeper,
    x: usize,
    y: usize,
    exact_on: bool,
    exact: Option<f32>,
    calculating: bool,
    guess: Option<f32>,
    over: bool,
    show_probs: bool,
) -> CellView {
    let cell = &game.grid[y][x];
    match cell.state {
        CellState::Hidden | CellState::Flagged => {
            // The tint follows whichever estimate is on screen. While the exact
            // answer is still coming, the cell keeps plain grey rather than a
            // colour belonging to the previous board — or to no board at all.
            let bg = match (exact_on, exact, guess) {
                (true, Some(p), _) => prob_color(p),
                (true, None, _) if !calculating => UNKNOWN.to_string(),
                (false, _, Some(g)) => prob_color(g),
                _ => "#cccccc".to_string(),
            };
            let labels = show_probs && !over;
            let prob = match (labels && exact_on, exact) {
                (true, Some(p)) => format!("{:.0}%", p * 100.0),
                (true, None) if !calculating => "?".to_string(),
                _ => String::new(),
            };
            let guess = match (labels, guess) {
                (true, Some(g)) => format!("{:.0}%", g * 100.0),
                _ => String::new(),
            };
            let flagged = cell.state == CellState::Flagged;
            CellView {
                text: if flagged { "⚑".into() } else { String::new() },
                fg: "#c1121f".into(),
                bg,
                raised: true,
                border: next_to_number(game, x, y),
                prob,
                guess,
            }
        }
        CellState::Visible => match cell.content {
            CellContent::Mine => CellView {
                text: "✹".into(),
                fg: "#ffffff".into(),
                bg: "#e5383b".into(),
                ..CellView::default()
            },
            CellContent::Empty(n) => CellView {
                text: if n > 0 { n.to_string() } else { String::new() },
                fg: number_color(n).into(),
                bg: "#eeeeee".into(),
                ..CellView::default()
            },
        },
    }
}

/// Whether an unopened cell touches a visible number.
fn next_to_number(game: &Minesweeper, x: usize, y: usize) -> bool {
    (y.saturating_sub(1)..=(y + 1).min(game.height - 1)).any(|ny| {
        (x.saturating_sub(1)..=(x + 1).min(game.width - 1)).any(|nx| {
            let n = &game.grid[ny][nx];
            n.state == CellState::Visible && matches!(n.content, CellContent::Empty(k) if k > 0)
        })
    })
}

/// Grey → red, matching the page, the CLI and the Actix front-end.
fn prob_color(p: f32) -> String {
    let r = (204.0 + 51.0 * p).round() as u8;
    let gb = (204.0 * (1.0 - p)).round() as u8;
    format!("#{r:02x}{gb:02x}{gb:02x}")
}

/// The classic digit colours, as `.n1`–`.n8` on the page.
fn number_color(n: u8) -> &'static str {
    match n {
        1 => "#0000ff",
        2 => "#007b00",
        3 => "#e00000",
        4 => "#00007b",
        5 => "#7b0000",
        6 => "#008080",
        7 => "#000000",
        _ => "#808080",
    }
}

fn fmt_memory(bytes: usize) -> String {
    match bytes {
        b if b < 1_024 => format!("{b} B"),
        b if b < 1_024 * 1_024 => format!("{:.1} KB", b as f64 / 1_024.0),
        b => format!("{:.1} MB", b as f64 / 1_048_576.0),
    }
}

/// `1234567` → `1,234,567`, as `toLocaleString` gives the page.
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn main() {
    qml_register_type::<MinesweeperGui>(cstr!("Minesweeper"), 1, 0, cstr!("MinesweeperGame"));
    let mut engine = QmlEngine::new();
    engine.load_data(include_str!("main.qml").into());
    engine.exec();
}
