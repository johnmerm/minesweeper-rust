use crossterm::cursor;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::style::{Color, ResetColor, SetBackgroundColor, SetForegroundColor};
use crossterm::terminal::{self, Clear, ClearType, enable_raw_mode, disable_raw_mode};
use minesweeper_core::{Minesweeper, CellState, CellContent, GameState};
use minesweeper_core::probability::{certain_cells, ConstraintSearch, MonteCarlo, SimUpdate};
use std::io::{stdout, Write};
use std::sync::mpsc::Sender;

fn prob_to_bg(p: f64) -> Color {
    let r = (204.0 + 51.0 * p).round() as u8;
    let g = (204.0 * (1.0 - p)).round() as u8;
    Color::Rgb { r, g, b: g }
}

/// Run a strategy synchronously, returning (probs, valid, attempts, memory_bytes).
fn run_sync(run: impl FnOnce(Sender<SimUpdate>)) -> (Vec<Vec<f64>>, usize, usize, usize) {
    let (tx, rx) = std::sync::mpsc::channel();
    run(tx);
    let mut probs = Vec::new();
    let mut valid = 0;
    let mut attempts = 0;
    let mut mem = 0;
    while let Ok(update) = rx.recv() {
        if let SimUpdate::Done { probs: p, valid: v, attempts: a, memory_bytes: m, .. } = update {
            probs = p;
            valid = v;
            attempts = a;
            mem = m;
            break;
        }
    }
    (probs, valid, attempts, mem)
}

fn fmt_memory(bytes: usize) -> String {
    match bytes {
        b if b < 1_024          => format!("{} B", b),
        b if b < 1_024 * 1_024  => format!("{:.1} KB", b as f64 / 1_024.0),
        b                       => format!("{:.1} MB", b as f64 / 1_048_576.0),
    }
}

/// Returns the probabilities, whether they are exact rather than sampled, and the
/// two status lines.
///
/// The exactness flag is not cosmetic: a sampled 0% only means no draw happened
/// to put a mine there, so auto-reveal must not act on it.
fn compute_probs(
    exact: &ConstraintSearch,
    game: &Minesweeper,
) -> (Vec<Vec<f64>>, bool, String, String) {
    let (cs_probs, cs_valid, cs_attempts, cs_mem) =
        run_sync(|tx| exact.calculate_with_progress(game, tx));

    // Sampling only when the exact search comes back with nothing. It is slower
    // and less accurate, so running it every time was work thrown away.
    let (mc_probs, mc_valid, mc_attempts, mc_mem) = if cs_valid > 0 {
        (Vec::new(), 0, 0, 0)
    } else {
        run_sync(|tx| MonteCarlo::new().calculate_with_progress(game, tx))
    };

    let probs = if cs_valid > 0 { cs_probs } else { mc_probs };
    let mc_status = format!(
        "MC: {} valid / {} sampled  [{}]",
        mc_valid, mc_attempts, fmt_memory(mc_mem)
    );
    let cs_status = format!(
        "CS: {} layouts / {} steps  [{}]",
        cs_valid, cs_attempts, fmt_memory(cs_mem)
    );
    (probs, cs_valid > 0, mc_status, cs_status)
}

/// Open every cell that can be proven safe, and flag every cell proven to be a
/// mine. Returns true if anything was opened.
///
/// Iterates on constraint propagation, which proves what local rules can with no
/// search at all, and only consults `probs` — one full solve, already paid for by
/// the caller — once propagation has run dry. Re-solving after every pass is what
/// made this take half a minute on a dense board.
fn apply_auto_reveal(game: &mut Minesweeper, probs: &[Vec<f64>], probs_are_exact: bool) -> bool {
    if game.state != GameState::Playing || !game.mines_generated {
        return false;
    }

    let mut opened = 0;
    loop {
        let proven = certain_cells(game);
        let mut this_pass = 0;
        for (x, y) in proven.safe {
            if game.grid[y][x].state == CellState::Hidden {
                game.reveal(x, y);
                this_pass += 1;
            }
        }
        for (x, y) in proven.mines {
            if game.grid[y][x].state == CellState::Hidden {
                game.toggle_flag(x, y);
            }
        }
        opened += this_pass;
        if this_pass == 0 || game.state != GameState::Playing {
            break;
        }
    }

    // A sampled 0% means only that no draw happened to place a mine there, so it
    // is never grounds for opening a cell.
    if opened == 0 && probs_are_exact && game.state == GameState::Playing {
        let safe: Vec<(usize, usize)> = (0..game.height)
            .flat_map(|y| (0..game.width).map(move |x| (x, y)))
            .filter(|&(x, y)| game.grid[y][x].state == CellState::Hidden && probs[y][x] < 1e-9)
            .collect();
        for (x, y) in safe {
            game.reveal(x, y);
            opened += 1;
        }
    }

    opened > 0
}

fn parse_args() -> (usize, usize, usize) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let w = args.get(0).and_then(|s| s.parse::<usize>().ok()).unwrap_or(10);
    let h = args.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(10);
    let m = args.get(2).and_then(|s| s.parse::<usize>().ok()).unwrap_or(10);
    let w = w.clamp(3, 50);
    let h = h.clamp(3, 50);
    let m = m.clamp(1, w * h - 1);
    (w, h, m)
}

fn main() -> std::io::Result<()> {
    let (init_w, init_h, init_m) = parse_args();
    let mut game = Minesweeper::new(init_w, init_h, init_m);
    let mut cursor_x = 0usize;
    let mut cursor_y = 0usize;
    let mut auto_reveal = false;
    // One solver for the whole session: it caches region solutions between moves.
    let exact = ConstraintSearch::new();
    let (mut probs, mut probs_exact, mut mc_status, mut cs_status) = compute_probs(&exact, &game);

    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, terminal::EnterAlternateScreen, cursor::Hide)?;

    loop {
        execute!(stdout, Clear(ClearType::All), cursor::MoveTo(0, 0))?;

        let ar_label = if auto_reveal { "ON" } else { "OFF" };
        println!("Minesweeper - Arrows: move  Space: reveal  F: flag  A: auto-reveal({})  R: restart  Q: quit\r", ar_label);
        println!("{}x{} board  |  Mines: {}\r", game.width, game.height, game.mines_count);
        println!("\r");

        for y in 0..game.height {
            for x in 0..game.width {
                let cell = &game.grid[y][x];

                if x == cursor_x && y == cursor_y {
                    execute!(stdout, SetBackgroundColor(Color::White), SetForegroundColor(Color::Black))?;
                } else if matches!(cell.state, CellState::Hidden | CellState::Flagged) {
                    execute!(stdout, SetBackgroundColor(prob_to_bg(probs[y][x])))?;
                }

                let symbol = match cell.state {
                    CellState::Hidden => " . ".to_string(),
                    CellState::Flagged => " F ".to_string(),
                    CellState::Visible => match cell.content {
                        CellContent::Mine => " * ".to_string(),
                        CellContent::Empty(0) => "   ".to_string(),
                        CellContent::Empty(n) => format!(" {} ", n),
                    },
                };

                if cell.state == CellState::Visible {
                    if let CellContent::Empty(n) = cell.content {
                        match n {
                            1 => { execute!(stdout, SetForegroundColor(Color::Blue))?; }
                            2 => { execute!(stdout, SetForegroundColor(Color::Green))?; }
                            3 => { execute!(stdout, SetForegroundColor(Color::Red))?; }
                            _ => {}
                        }
                    }
                    if let CellContent::Mine = cell.content {
                        execute!(stdout, SetForegroundColor(Color::Red))?;
                    }
                }

                if x == cursor_x && y == cursor_y {
                } else if cell.state == CellState::Flagged {
                    execute!(stdout, SetForegroundColor(Color::Yellow))?;
                }

                print!("{}", symbol);
                execute!(stdout, ResetColor)?;
            }
            println!("\r");
        }

        // Strategy stats
        execute!(stdout, SetForegroundColor(Color::DarkGrey))?;
        println!("{}\r", mc_status);
        execute!(stdout, SetForegroundColor(Color::Rgb { r: 102, g: 102, b: 136 }))?;
        println!("{}\r", cs_status);
        execute!(stdout, ResetColor)?;

        if game.state == GameState::Won {
            println!("\r\nYOU WON! Press Q to quit.\r");
        } else if game.state == GameState::Lost {
            println!("\r\nGAME OVER! Press Q to quit.\r");
        } else if matches!(game.grid[cursor_y][cursor_x].state, CellState::Hidden | CellState::Flagged) {
            println!("Mine probability here: {:.1}%\r", probs[cursor_y][cursor_x] * 100.0);
        }

        stdout.flush()?;

        if event::poll(std::time::Duration::from_millis(500))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Up => if cursor_y > 0 { cursor_y -= 1; },
                        KeyCode::Down => if cursor_y < game.height - 1 { cursor_y += 1; },
                        KeyCode::Left => if cursor_x > 0 { cursor_x -= 1; },
                        KeyCode::Right => if cursor_x < game.width - 1 { cursor_x += 1; },
                        KeyCode::Char(' ') => {
                            if game.state == GameState::Playing {
                                game.reveal(cursor_x, cursor_y);
                                (probs, probs_exact, mc_status, cs_status) = compute_probs(&exact, &game);
                                if auto_reveal {
                                    while apply_auto_reveal(&mut game, &probs, probs_exact) {
                                        (probs, probs_exact, mc_status, cs_status) = compute_probs(&exact, &game);
                                    }
                                }
                            }
                        },
                        KeyCode::Char('f') => {
                            if game.state == GameState::Playing {
                                game.toggle_flag(cursor_x, cursor_y);
                                (probs, probs_exact, mc_status, cs_status) = compute_probs(&exact, &game);
                                if auto_reveal {
                                    while apply_auto_reveal(&mut game, &probs, probs_exact) {
                                        (probs, probs_exact, mc_status, cs_status) = compute_probs(&exact, &game);
                                    }
                                }
                            }
                        },
                        KeyCode::Char('a') => {
                            auto_reveal = !auto_reveal;
                            // Apply immediately if turned on mid-game.
                            if auto_reveal {
                                while apply_auto_reveal(&mut game, &probs, probs_exact) {
                                    (probs, probs_exact, mc_status, cs_status) = compute_probs(&exact, &game);
                                }
                            }
                        },
                        KeyCode::Char('r') => {
                            game = Minesweeper::new(init_w, init_h, init_m);
                            cursor_x = 0;
                            cursor_y = 0;
                            (probs, probs_exact, mc_status, cs_status) = compute_probs(&exact, &game);
                        },
                        _ => {}
                    }
                }
            }
        }
    }

    execute!(stdout, cursor::Show, terminal::LeaveAlternateScreen)?;
    disable_raw_mode()?;
    Ok(())
}
