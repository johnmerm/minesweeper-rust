//! Terminal front-end: the same game, estimators and wording as the page in
//! `docs/`, drawn with crossterm.
//!
//! The exact search and the network run on a [`session`] thread, as they do
//! behind the Qt GUI and the web server; this loop only reads keys, drains the
//! session's replies into the shared [`Controller`], and redraws when something
//! changed. A key never waits on a solve.
//!
//! A terminal cell holds one label, so where the page shows both estimates in
//! each square this shows the one the tint follows — the exact value, or the
//! network's in *neural network only* — and puts both on the line for the cell
//! under the cursor.

use crossterm::cursor;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetBackgroundColor, SetForegroundColor};
use crossterm::terminal::{self, disable_raw_mode, enable_raw_mode, Clear, ClearType};
use crossterm::{execute, queue};
use minesweeper_core::session::{self, Controller, Exact, Reply, Show};
use minesweeper_core::{CellContent, CellState, GameState};
use std::io::{stdout, Stdout, Write};
use std::sync::mpsc::Receiver;
use std::time::Duration;

/// Columns per board cell.
const CELL_W: u16 = 3;
/// Lines above the board: title, status, blank.
const HEADER_LINES: u16 = 3;
/// Lines below it: solver, network, cursor cell, blank, two of help.
const FOOTER_LINES: u16 = 6;

/// How long to wait for a key before looking at the session again. Short enough
/// that answers appear as they arrive, long enough to cost nothing.
const TICK: Duration = Duration::from_millis(50);

const PRESETS: [(usize, usize, usize); 3] = [(9, 9, 10), (16, 16, 40), (30, 16, 99)];

/// The page's colours.
const OPENED: Color = rgb(0xee, 0xee, 0xee);
const MINE: Color = rgb(0xe5, 0x38, 0x3b);
const CALCULATING: Color = rgb(0xcc, 0xcc, 0xcc);
/// A cell on a board the solver could not finish. Deliberately not the grey that
/// `prob_color(0.0)` produces, which is what "certainly safe" looks like.
const UNKNOWN: Color = rgb(0xc6, 0xcb, 0xd6);
const FLAG: Color = rgb(0xc1, 0x12, 0x1f);
const WON: Color = rgb(0x1a, 0x7f, 0x37);
const LOST: Color = rgb(0xc1, 0x12, 0x1f);
const MUTED: Color = rgb(0x88, 0x88, 0x88);
const SOLVED: Color = rgb(0x66, 0x66, 0x99);
const REFUSED: Color = rgb(0xa3, 0x52, 0x1b);
const NETWORK: Color = rgb(0x8a, 0x6f, 0xd6);

const fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color::Rgb { r, g, b }
}

/// Grey → red, matching the page, the Qt GUI and the web server.
fn prob_color(p: f32) -> Color {
    let r = (204.0 + 51.0 * p).round() as u8;
    let gb = (204.0 * (1.0 - p)).round() as u8;
    rgb(r, gb, gb)
}

/// The classic digit colours, as `.n1`–`.n8` on the page.
fn number_color(n: u8) -> Color {
    match n {
        1 => rgb(0x00, 0x00, 0xff),
        2 => rgb(0x00, 0x7b, 0x00),
        3 => rgb(0xe0, 0x00, 0x00),
        4 => rgb(0x00, 0x00, 0x7b),
        5 => rgb(0x7b, 0x00, 0x00),
        6 => rgb(0x00, 0x80, 0x80),
        7 => rgb(0x00, 0x00, 0x00),
        _ => rgb(0x80, 0x80, 0x80),
    }
}

/// `cli [width height mines]`, defaulting to the page's opening 10x10 with 10.
fn parse_args() -> (usize, usize, usize) {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    match args[..] {
        [w, h, m, ..] => (w, h, m),
        _ => (10, 10, 10),
    }
}

struct App {
    ctl: Controller,
    replies: Receiver<Reply>,
    cursor: (usize, usize),
    /// Top-left board cell on screen, for boards larger than the terminal.
    scroll: (usize, usize),
    show_probs: bool,
    /// Something changed since the last draw.
    dirty: bool,
    /// The clock's last drawn second, so a running clock redraws once a second.
    drawn_second: u64,
}

impl App {
    fn new(width: usize, height: usize, mines: usize) -> Self {
        let (jobs, inbox) = std::sync::mpsc::channel();
        let (reply_tx, replies) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            session::run(inbox, move |reply| {
                let _ = reply_tx.send(reply);
            })
        });
        let mut ctl = Controller::new(jobs);
        ctl.new_game(width, height, mines);
        Self {
            ctl,
            replies,
            cursor: (0, 0),
            scroll: (0, 0),
            show_probs: true,
            dirty: true,
            drawn_second: 0,
        }
    }

    /// Take in whatever the session has said since the last look.
    fn drain(&mut self) {
        while let Ok(reply) = self.replies.try_recv() {
            self.dirty |= self.ctl.on_reply(reply);
        }
        let second = self.ctl.elapsed().as_secs();
        if second != self.drawn_second {
            self.drawn_second = second;
            self.dirty = true;
        }
    }

    fn new_game(&mut self, width: usize, height: usize, mines: usize) {
        self.ctl.new_game(width, height, mines);
        self.cursor = (0, 0);
        self.scroll = (0, 0);
        self.dirty = true;
    }

    /// Returns false to quit.
    fn key(&mut self, code: KeyCode, modifiers: KeyModifiers) -> bool {
        let (w, h) = (self.ctl.game.width, self.ctl.game.height);
        let (x, y) = self.cursor;
        match code {
            KeyCode::Char('q') | KeyCode::Esc => return false,
            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => return false,
            KeyCode::Up | KeyCode::Char('k') => self.cursor.1 = y.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => self.cursor.1 = (y + 1).min(h - 1),
            KeyCode::Left | KeyCode::Char('h') => self.cursor.0 = x.saturating_sub(1),
            KeyCode::Right | KeyCode::Char('l') => self.cursor.0 = (x + 1).min(w - 1),
            KeyCode::Home => self.cursor.0 = 0,
            KeyCode::End => self.cursor.0 = w - 1,
            KeyCode::PageUp => self.cursor.1 = 0,
            KeyCode::PageDown => self.cursor.1 = h - 1,
            KeyCode::Char(' ') | KeyCode::Enter => {
                self.ctl.reveal(x, y);
            }
            KeyCode::Char('f') => {
                self.ctl.flag(x, y);
            }
            KeyCode::Char('a') => {
                let on = !self.ctl.auto_play;
                self.ctl.set_auto_play(on);
            }
            KeyCode::Char('s') => {
                let next = Show::from_index((self.ctl.show.index() + 1) % 3);
                self.ctl.set_show(next);
            }
            KeyCode::Char('p') => self.show_probs = !self.show_probs,
            KeyCode::Char('n') => {
                let game = &self.ctl.game;
                let size = (game.width, game.height, game.mines_count);
                self.new_game(size.0, size.1, size.2);
            }
            KeyCode::Char(c @ '1'..='3') => {
                let (w, h, m) = PRESETS[c as usize - '1' as usize];
                self.new_game(w, h, m);
            }
            _ => return true,
        }
        self.dirty = true;
        true
    }

    /// Keep the cursor on screen, scrolling by as little as possible.
    fn follow_cursor(&mut self, cols: usize, rows: usize) {
        let follow = |scroll: usize, at: usize, span: usize| {
            if at < scroll {
                at
            } else if at >= scroll + span {
                at + 1 - span
            } else {
                scroll
            }
        };
        self.scroll.0 = follow(self.scroll.0, self.cursor.0, cols.max(1));
        self.scroll.1 = follow(self.scroll.1, self.cursor.1, rows.max(1));
    }

    fn draw(&mut self, out: &mut Stdout) -> std::io::Result<()> {
        let (term_w, term_h) = terminal::size()?;
        let game = &self.ctl.game;
        let cols = (term_w / CELL_W) as usize;
        let rows = term_h.saturating_sub(HEADER_LINES + FOOTER_LINES) as usize;
        let cols = cols.min(game.width);
        let rows = rows.min(game.height);
        self.follow_cursor(cols, rows);
        let ctl = &self.ctl;
        let game = &ctl.game;

        // Overwrite in place, clearing each line's tail, rather than clearing the
        // screen first: the network reports several times a second, and a full
        // clear per report flickers.
        queue!(out, cursor::MoveTo(0, 0))?;

        // Title and the scoreboard.
        queue!(out, SetAttribute(Attribute::Bold), Print("Minesweeper"), SetAttribute(Attribute::Reset))?;
        let flags = game.grid.iter().flatten().filter(|c| c.state == CellState::Flagged).count();
        let secs = ctl.elapsed().as_secs();
        queue!(
            out,
            SetForegroundColor(MUTED),
            Print(format!("   {}x{}   Mines ", game.width, game.height)),
            ResetColor,
            Print(game.mines_count as i64 - flags as i64),
            SetForegroundColor(MUTED),
            Print("   Time "),
            ResetColor,
            Print(format!("{}:{:02}", secs / 60, secs % 60)),
            Clear(ClearType::UntilNewLine), cursor::MoveToNextLine(1),
        )?;
        let (status, colour) = match game.state {
            GameState::Playing => ("Playing", None),
            GameState::Won => ("You won!", Some(WON)),
            GameState::Lost => ("Boom — game over", Some(LOST)),
        };
        queue!(out, SetAttribute(Attribute::Bold))?;
        if let Some(colour) = colour {
            queue!(out, SetForegroundColor(colour))?;
        }
        queue!(out, Print(status), ResetColor, SetAttribute(Attribute::Reset), Clear(ClearType::UntilNewLine), cursor::MoveToNextLine(1), Clear(ClearType::UntilNewLine), cursor::MoveToNextLine(1))?;

        // The board.
        let exact_on = ctl.show != Show::Neural;
        let neural_on = ctl.neural_wanted();
        let solved = ctl.exact.as_ref().and_then(|e| e.probs.as_ref());
        let calculating = ctl.exact.is_none();
        let over = game.state != GameState::Playing;
        for y in self.scroll.1..self.scroll.1 + rows {
            for x in self.scroll.0..self.scroll.0 + cols {
                let i = y * game.width + x;
                let exact = solved.map(|p| p[i]);
                let guess = ctl.neural.probs.get(i).copied().filter(|&g| neural_on && g >= 0.0);
                let (text, fg, bg) = cell_look(game, x, y, exact_on, exact, calculating, guess, over, self.show_probs);
                if (x, y) == self.cursor {
                    // Inverted, so the cell under the cursor keeps its colour as text.
                    queue!(out, SetBackgroundColor(Color::Black), SetForegroundColor(bg), SetAttribute(Attribute::Bold))?;
                } else {
                    queue!(out, SetBackgroundColor(bg), SetForegroundColor(fg))?;
                }
                queue!(out, Print(text), SetAttribute(Attribute::Reset), ResetColor)?;
            }
            queue!(out, Clear(ClearType::UntilNewLine), cursor::MoveToNextLine(1))?;
        }
        if cols < game.width || rows < game.height {
            queue!(
                out,
                SetForegroundColor(MUTED),
                Print(format!(
                    "showing columns {}–{} of {}, rows {}–{} of {}",
                    self.scroll.0 + 1,
                    self.scroll.0 + cols,
                    game.width,
                    self.scroll.1 + 1,
                    self.scroll.1 + rows,
                    game.height
                )),
                ResetColor,
            )?;
        }
        queue!(out, Clear(ClearType::UntilNewLine), cursor::MoveToNextLine(1))?;

        // What the solver and the network did, and the cell under the cursor.
        let (sim, sim_colour) = describe_exact(ctl.exact.as_ref());
        queue!(out, SetForegroundColor(sim_colour), Print(sim), ResetColor, Clear(ClearType::UntilNewLine), cursor::MoveToNextLine(1))?;
        queue!(out, SetForegroundColor(NETWORK), Print(describe_neural(ctl)), ResetColor, Clear(ClearType::UntilNewLine), cursor::MoveToNextLine(1))?;
        queue!(out, Print(self.cursor_line()), Clear(ClearType::UntilNewLine), cursor::MoveToNextLine(1), Clear(ClearType::UntilNewLine), cursor::MoveToNextLine(1))?;

        let on_off = |on: bool| if on { "on" } else { "off" };
        let show = match ctl.show {
            Show::Both => "both estimates",
            Show::Exact => "constraint search only",
            Show::Neural => "neural network only",
        };
        queue!(
            out,
            SetForegroundColor(MUTED),
            Print("arrows/hjkl move · space reveal · f flag · n new · 1/2/3 beginner/intermediate/expert · q quit"),
            Clear(ClearType::UntilNewLine), cursor::MoveToNextLine(1),
            Print(format!(
                "a auto-play deductions ({}) · s show: {show} · p probabilities ({})",
                on_off(ctl.auto_play),
                on_off(self.show_probs)
            )),
            ResetColor,
            Clear(ClearType::FromCursorDown),
        )?;
        out.flush()?;
        self.dirty = false;
        Ok(())
    }

    /// The line for the cell under the cursor, as the page's hover line.
    fn cursor_line(&self) -> String {
        let ctl = &self.ctl;
        let (x, y) = self.cursor;
        if ctl.game.grid[y][x].state == CellState::Visible {
            return String::new();
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
        parts.join(" · ")
    }
}

/// How one cell looks: its three columns of text, and their colours.
///
/// The tint follows whichever estimate is on screen. While the exact answer is
/// still coming the cell stays plain grey, never a colour from the previous
/// board; once the search has refused, it takes the `UNKNOWN` tint and a `?`.
#[allow(clippy::too_many_arguments)]
fn cell_look(
    game: &minesweeper_core::Minesweeper,
    x: usize,
    y: usize,
    exact_on: bool,
    exact: Option<f32>,
    calculating: bool,
    guess: Option<f32>,
    over: bool,
    show_probs: bool,
) -> (String, Color, Color) {
    let cell = &game.grid[y][x];
    match cell.state {
        CellState::Hidden | CellState::Flagged => {
            let shown = if exact_on { exact } else { guess };
            let bg = match shown {
                Some(p) => prob_color(p),
                None if exact_on && !calculating => UNKNOWN,
                None => CALCULATING,
            };
            if cell.state == CellState::Flagged {
                return (" F ".into(), FLAG, bg);
            }
            let label = match shown {
                _ if !show_probs || over => String::new(),
                Some(p) => format!("{:.0}", p * 100.0),
                None if exact_on && !calculating => "?".into(),
                None => String::new(),
            };
            (format!("{label:>3}"), rgb(0x44, 0x44, 0x44), bg)
        }
        CellState::Visible => match cell.content {
            CellContent::Mine => (" * ".into(), Color::White, MINE),
            CellContent::Empty(0) => ("   ".into(), Color::Black, OPENED),
            CellContent::Empty(n) => (format!(" {n} "), number_color(n), OPENED),
        },
    }
}

/// The solver line, worded as `renderSim` in `docs/minesweeper.js`.
fn describe_exact(exact: Option<&Exact>) -> (String, Color) {
    let Some(exact) = exact else { return ("calculating…".into(), MUTED) };
    if exact.probs.is_none() {
        let text = format!(
            "no exact answer within {} nodes — showing none rather than guessing",
            thousands(exact.nodes)
        );
        return (text, REFUSED);
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
    (text, SOLVED)
}

/// The network line, worded as `describeNeural` in `docs/minesweeper.js`.
fn describe_neural(ctl: &Controller) -> String {
    let neural = &ctl.neural;
    if neural.broken {
        return "network: this build carries no usable weights".into();
    }
    if ctl.show != Show::Exact && !ctl.neural_wanted() {
        return "network: not scored automatically on a board this large — press s to ask for it".into();
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

fn run(out: &mut Stdout) -> std::io::Result<()> {
    let (w, h, m) = parse_args();
    let mut app = App::new(w, h, m);
    loop {
        app.drain();
        if app.dirty {
            app.draw(out)?;
        }
        if event::poll(TICK)? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if !app.key(key.code, key.modifiers) {
                        return Ok(());
                    }
                }
                Event::Resize(..) => app.dirty = true,
                _ => {}
            }
        }
    }
}

fn main() -> std::io::Result<()> {
    enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, terminal::EnterAlternateScreen, cursor::Hide)?;
    // Restore the terminal however `run` ends, error included.
    let result = run(&mut out);
    execute!(out, ResetColor, cursor::Show, terminal::LeaveAlternateScreen)?;
    disable_raw_mode()?;
    result
}
