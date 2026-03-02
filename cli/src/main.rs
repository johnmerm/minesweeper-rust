use crossterm::cursor;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::style::{Color, ResetColor, SetBackgroundColor, SetForegroundColor};
use crossterm::terminal::{self, Clear, ClearType, enable_raw_mode, disable_raw_mode};
use minesweeper_core::{Minesweeper, CellState, CellContent, GameState};
use minesweeper_core::probability::{MonteCarlo, ConstraintSearch, SimUpdate};
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

fn compute_probs(game: &Minesweeper) -> (Vec<Vec<f64>>, String, String) {
    let (mc_probs, mc_valid, mc_attempts, mc_mem) =
        run_sync(|tx| MonteCarlo::new().calculate_with_progress(game, tx));
    let (cs_probs, cs_valid, cs_attempts, cs_mem) =
        run_sync(|tx| ConstraintSearch::new().calculate_with_progress(game, tx));

    let probs = if cs_valid > 0 { cs_probs } else { mc_probs };
    let mc_status = format!(
        "MC: {} valid / {} sampled  [{}]",
        mc_valid, mc_attempts, fmt_memory(mc_mem)
    );
    let cs_status = format!(
        "CS: {} layouts / {} steps  [{}]",
        cs_valid, cs_attempts, fmt_memory(cs_mem)
    );
    (probs, mc_status, cs_status)
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
    let (mut probs, mut mc_status, mut cs_status) = compute_probs(&game);

    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, terminal::EnterAlternateScreen, cursor::Hide)?;

    loop {
        execute!(stdout, Clear(ClearType::All), cursor::MoveTo(0, 0))?;

        println!("Minesweeper - Arrows: move  Space: reveal  F: flag  R: restart  Q: quit\r");
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
                                (probs, mc_status, cs_status) = compute_probs(&game);
                            }
                        },
                        KeyCode::Char('f') => {
                            if game.state == GameState::Playing {
                                game.toggle_flag(cursor_x, cursor_y);
                                (probs, mc_status, cs_status) = compute_probs(&game);
                            }
                        },
                        KeyCode::Char('r') => {
                            game = Minesweeper::new(init_w, init_h, init_m);
                            cursor_x = 0;
                            cursor_y = 0;
                            (probs, mc_status, cs_status) = compute_probs(&game);
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
