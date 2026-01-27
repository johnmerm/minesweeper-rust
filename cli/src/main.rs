use crossterm::cursor;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::style::{Color, Print, ResetColor, SetBackgroundColor, SetForegroundColor};
use crossterm::terminal::{self, Clear, ClearType, enable_raw_mode, disable_raw_mode};
use minesweeper_core::{Minesweeper, CellState, CellContent, GameState};
use std::io::{stdout, Write};

fn main() -> std::io::Result<()> {
    let mut game = Minesweeper::new(10, 10, 10);
    let mut cursor_x = 0;
    let mut cursor_y = 0;

    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, terminal::EnterAlternateScreen, cursor::Hide)?;

    loop {
        execute!(stdout, Clear(ClearType::All), cursor::MoveTo(0, 0))?;

        println!("Minesweeper - Arrow Keys to move, Space to reveal, F to flag, Q to quit\r");
        println!("Mines: {}\r", game.mines_count);
        println!("\r");

        for y in 0..game.height {
            for x in 0..game.width {
                let cell = &game.grid[y][x];
                
                if x == cursor_x && y == cursor_y {
                    execute!(stdout, SetBackgroundColor(Color::White), SetForegroundColor(Color::Black))?;
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

        if game.state == GameState::Won {
             println!("\r\nYOU WON! Press Q to quit.\r");
        } else if game.state == GameState::Lost {
             println!("\r\nGAME OVER! Press Q to quit.\r");
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
                            }
                        },
                        KeyCode::Char('f') => {
                             if game.state == GameState::Playing {
                                game.toggle_flag(cursor_x, cursor_y);
                            }
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