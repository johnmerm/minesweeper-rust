//! The 9x9 view of the board that the neural estimators are fed.
//!
//! One cell's patch is what either network sees of the world: the squares around
//! it, what is known about each, and one global number. The layout is a contract
//! between three places — this file, `neural/dataset.py` which builds the same
//! patches for training, and whatever weights came out of that training. Get a
//! channel wrong in one of them and the model is served inputs it never saw, which
//! produces confident nonsense rather than an error.
//!
//! It lives here, outside the `neural` feature and outside either estimator, so
//! that at least the two Rust readers cannot drift apart.
//!
//! | Ch | Value                          | Description                        |
//! |----|--------------------------------|------------------------------------|
//! | 0  | mine_count / 8.0 if Visible    | Numbered cell value                |
//! | 1  | {0,1}                          | Is patch cell Visible              |
//! | 2  | {0,1}                          | Is patch cell Hidden               |
//! | 3  | {0,1}                          | Is patch cell Flagged              |
//! | 4  | {0,1}                          | Is patch cell out-of-bounds        |
//! | 5  | 1.0 at (4,4) else 0            | Target cell marker                 |
//! | 6  | mines_remaining / total_hidden | Global ratio broadcast             |
//! | 7  | {0,1}                          | Is patch cell a border hidden cell |

use crate::{CellContent, CellState, Minesweeper};

pub(crate) const HALF: usize = 4;
pub(crate) const PATCH: usize = 2 * HALF + 1; // 9
pub(crate) const N_CHANNELS: usize = 8;
/// Values in one patch.
pub(crate) const PATCH_LEN: usize = N_CHANNELS * PATCH * PATCH;

/// Everything about a board that patch extraction needs, computed once.
///
/// Built per board rather than per cell: the border mask in particular is a
/// whole-board property, and recomputing it for each of a few hundred cells is
/// what made the Python dataset unusable.
pub(crate) struct PatchSource {
    width: usize,
    height: usize,
    /// 0 hidden, 1 visible, 2 flagged.
    state: Vec<u8>,
    /// 0..8 for a visible number, 9 for a visible mine, 255 otherwise.
    content: Vec<u8>,
    /// Unopened cells touching a visible number.
    border: Vec<bool>,
    /// Mines left over the cells still unopened.
    pub(crate) mines_ratio: f32,
}

impl PatchSource {
    pub(crate) fn new(game: &Minesweeper) -> Self {
        let (width, height) = (game.width, game.height);
        let mut state = vec![0u8; width * height];
        let mut content = vec![255u8; width * height];
        let mut hidden = 0usize;
        let mut visible_mines = 0usize;

        for y in 0..height {
            for x in 0..width {
                let cell = &game.grid[y][x];
                let at = y * width + x;
                state[at] = match cell.state {
                    CellState::Hidden => 0,
                    CellState::Visible => 1,
                    CellState::Flagged => 2,
                };
                if cell.state == CellState::Visible {
                    content[at] = match cell.content {
                        CellContent::Empty(n) => n,
                        CellContent::Mine => {
                            visible_mines += 1;
                            9
                        }
                    };
                }
                if cell.state == CellState::Hidden {
                    hidden += 1;
                }
            }
        }

        let mut border = vec![false; width * height];
        for y in 0..height {
            for x in 0..width {
                let at = y * width + x;
                if state[at] == 1 {
                    continue; // only unopened cells can be border cells
                }
                border[at] = neighbours(x, y, width, height).any(|(nx, ny)| {
                    let n = ny * width + nx;
                    state[n] == 1 && content[n] > 0 && content[n] < 9
                });
            }
        }

        let mines_ratio =
            game.mines_count.saturating_sub(visible_mines) as f32 / hidden.max(1) as f32;

        Self { width, height, state, content, border, mines_ratio }
    }

    /// Write the patch centred on `(cx, cy)` into `out`, which must hold
    /// [`PATCH_LEN`] values and start zeroed.
    ///
    /// Channel-major: `out[channel * PATCH * PATCH + row * PATCH + column]`.
    pub(crate) fn fill(&self, cx: usize, cy: usize, out: &mut [f32]) {
        debug_assert!(out.len() >= PATCH_LEN);
        let plane = PATCH * PATCH;

        for row in 0..PATCH {
            for column in 0..PATCH {
                let gy = cy as isize + row as isize - HALF as isize;
                let gx = cx as isize + column as isize - HALF as isize;
                let pos = row * PATCH + column;

                if gy < 0 || gy >= self.height as isize || gx < 0 || gx >= self.width as isize {
                    out[4 * plane + pos] = 1.0;
                    continue;
                }
                let at = gy as usize * self.width + gx as usize;
                match self.state[at] {
                    1 => {
                        out[plane + pos] = 1.0;
                        if self.content[at] < 9 {
                            out[pos] = self.content[at] as f32 / 8.0;
                        }
                    }
                    0 => out[2 * plane + pos] = 1.0,
                    2 => out[3 * plane + pos] = 1.0,
                    _ => {}
                }
                if self.border[at] {
                    out[7 * plane + pos] = 1.0;
                }
            }
        }

        out[5 * plane + HALF * PATCH + HALF] = 1.0;
        for pos in 0..plane {
            out[6 * plane + pos] = self.mines_ratio;
        }
    }
}

pub(super) fn neighbours(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let (w, h) = (width as isize, height as isize);
    (-1isize..=1)
        .flat_map(|dy| (-1isize..=1).map(move |dx| (dx, dy)))
        .filter(|&(dx, dy)| dx != 0 || dy != 0)
        .filter_map(move |(dx, dy)| {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            (nx >= 0 && nx < w && ny >= 0 && ny < h).then_some((nx as usize, ny as usize))
        })
}
