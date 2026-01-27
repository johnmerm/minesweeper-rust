use rand::Rng;
use std::fmt;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CellState {
    Hidden,
    Visible,
    Flagged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum CellContent {
    Empty(u8), // Number of neighboring mines
    Mine,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Cell {
    pub state: CellState,
    pub content: CellContent,
}

impl Cell {
    pub fn new() -> Self {
        Self {
            state: CellState::Hidden,
            content: CellContent::Empty(0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GameState {
    Playing,
    Won,
    Lost,
}

pub struct Minesweeper {
    pub width: usize,
    pub height: usize,
    pub mines_count: usize,
    pub grid: Vec<Vec<Cell>>,
    pub state: GameState,
    pub mines_generated: bool,
}

impl Minesweeper {
    pub fn new(width: usize, height: usize, mines_count: usize) -> Self {
        let grid = vec![vec![Cell::new(); width]; height];
        Self {
            width,
            height,
            mines_count,
            grid,
            state: GameState::Playing,
            mines_generated: false,
        }
    }

    fn generate_mines(&mut self, safe_x: usize, safe_y: usize) {
        let mut rng = rand::thread_rng();
        let mut placed = 0;

        while placed < self.mines_count {
            let x = rng.gen_range(0..self.width);
            let y = rng.gen_range(0..self.height);

            // Don't place mine on the first click or if already there
            if (x == safe_x && y == safe_y) || matches!(self.grid[y][x].content, CellContent::Mine) {
                continue;
            }

            self.grid[y][x].content = CellContent::Mine;
            placed += 1;
        }

        // Calculate numbers
        for y in 0..self.height {
            for x in 0..self.width {
                if matches!(self.grid[y][x].content, CellContent::Mine) {
                    continue;
                }

                let mines = self.count_neighbor_mines(x, y);
                self.grid[y][x].content = CellContent::Empty(mines);
            }
        }

        self.mines_generated = true;
    }

    fn count_neighbor_mines(&self, x: usize, y: usize) -> u8 {
        let mut count = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as isize + dx;
                let ny = y as isize + dy;

                if nx >= 0 && nx < self.width as isize && ny >= 0 && ny < self.height as isize {
                    if let CellContent::Mine = self.grid[ny as usize][nx as usize].content {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    pub fn reveal(&mut self, x: usize, y: usize) {
        if self.state != GameState::Playing {
            return;
        }

        if !self.mines_generated {
            self.generate_mines(x, y);
        }

        let cell = &mut self.grid[y][x];
        if cell.state != CellState::Hidden {
            return;
        }

        cell.state = CellState::Visible;

        if let CellContent::Mine = cell.content {
            self.state = GameState::Lost;
            self.reveal_all_mines();
            return;
        }

        if let CellContent::Empty(0) = cell.content {
            self.reveal_neighbors(x, y);
        }

        self.check_win();
    }

    fn reveal_neighbors(&mut self, x: usize, y: usize) {
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as isize + dx;
                let ny = y as isize + dy;

                if nx >= 0 && nx < self.width as isize && ny >= 0 && ny < self.height as isize {
                    let nx = nx as usize;
                    let ny = ny as usize;
                    if self.grid[ny][nx].state == CellState::Hidden {
                        // Recursively reveal
                         if let CellContent::Empty(0) = self.grid[ny][nx].content {
                             // Temporarily prevent infinite recursion by checking state first above
                             // But we need to call reveal
                             self.reveal(nx, ny);
                         } else {
                             // Just reveal this one if it's a number
                             self.grid[ny][nx].state = CellState::Visible;
                         }
                    }
                }
            }
        }
    }
    
    fn reveal_all_mines(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                if let CellContent::Mine = self.grid[y][x].content {
                     self.grid[y][x].state = CellState::Visible;
                }
            }
        }
    }

    pub fn toggle_flag(&mut self, x: usize, y: usize) {
         if self.state != GameState::Playing {
            return;
        }
        
        let cell = &mut self.grid[y][x];
        match cell.state {
            CellState::Hidden => cell.state = CellState::Flagged,
            CellState::Flagged => cell.state = CellState::Hidden,
            _ => {}
        }
    }

    fn check_win(&mut self) {
        let mut hidden_non_mines = 0;
        for y in 0..self.height {
            for x in 0..self.width {
                if let CellContent::Empty(_) = self.grid[y][x].content {
                    if self.grid[y][x].state != CellState::Visible {
                        hidden_non_mines += 1;
                    }
                }
            }
        }

        if hidden_non_mines == 0 {
            self.state = GameState::Won;
        }
    }
}

impl fmt::Display for Minesweeper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &self.grid {
            for cell in row {
                let c = match cell.state {
                    CellState::Hidden => ".",
                    CellState::Flagged => "F",
                    CellState::Visible => match cell.content {
                        CellContent::Mine => "*",
                        CellContent::Empty(0) => " ",
                        CellContent::Empty(n) => {
                            write!(f, "{} ", n)?;
                            continue;
                        }
                    },
                };
                write!(f, "{} ", c)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}