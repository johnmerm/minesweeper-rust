# Minesweeper – Project Overview & Development Guidelines

## Project Structure

This is a Rust workspace with a shared game engine and three independent front-ends.

```
minesweeper/
├── minesweeper_core/   # Game logic library (shared by all front-ends)
├── cli/                # Terminal UI (crossterm)
├── gui/                # Desktop GUI (Qt / qmetaobject QML)
└── web/                # Web server (Actix-Web + Tera templates)
```

### `minesweeper_core`

The single source of truth for all game state and logic.

| Item | Purpose |
|------|---------|
| `CellState` | `Hidden` / `Flagged` / `Visible` |
| `CellContent` | `Empty(u8)` (neighbour mine count) or `Mine` |
| `Cell` | Pairs a `CellState` with a `CellContent` |
| `GameState` | `Playing` / `Won` / `Lost` |
| `Minesweeper` | The board: `grid: Vec<Vec<Cell>>`, dimensions, mine count, game state |
| `Minesweeper::reveal` | Reveals a cell; generates mines lazily on the first call (safe-first-click guarantee) |
| `Minesweeper::toggle_flag` | Cycles `Hidden ↔ Flagged` |
| `Minesweeper::calculate_mine_probabilities` | Monte Carlo probability estimator – see below |

#### Mine probability estimator

`calculate_mine_probabilities(&self) -> Vec<Vec<f64>>`

Uses random sampling to estimate the probability that each unopened cell contains a mine:

1. Collects all `Hidden`/`Flagged` cells as candidates.
2. Builds constraints from every visible numbered cell: the count of mine-candidates among its neighbours must equal its displayed number (minus any already-visible mines).
3. Repeatedly draws a random subset of `mines_count` candidates (partial Fisher-Yates), validates all constraints, and tallies per-cell mine counts across valid draws.
4. Stops after **10 000 valid distributions** or **1 000 000 total attempts**, whichever comes first.
5. Returns `mine_count[cell] / valid_distributions` for each cell (0.0 for visible cells).

**Key constants** (top of `calculate_mine_probabilities`):

```rust
const MAX_VALID: usize = 10_000;
const MAX_ATTEMPTS: usize = 1_000_000;
```

Adjust these to trade accuracy for speed.

---

### `cli`

Terminal front-end using **crossterm 0.27**.

- Arrow keys move a cursor; `Space` reveals; `F` flags; `Q` quits.
- Probabilities are recomputed (via `calculate_mine_probabilities`) after every reveal or flag action.
- Unopened cells are coloured with an RGB background interpolated from grey `(204,204,204)` to red `(255,0,0)` based on mine probability.
- The status line below the grid shows the probability for the cell under the cursor.

### `gui`

Desktop front-end using **qmetaobject 0.2** (Qt 5 bindings) with an inline QML UI.

- Left-click reveals; right-click flags.
- `MinesweeperGui::update_view` converts the board + probabilities into a `QVariantList` of maps consumed by a QML `Repeater`.
- Each cell map carries: `text`, `color`, `bgColor` (probability-tinted for unopened cells), `probText` (e.g. `"23%"`).
- Hovering a cell updates the `hoveredProb` QML property, which replaces the mine-count in the status `Text` element.

### `web`

Web front-end using **Actix-Web 4** with **Tera** templates.

- Single shared `Mutex<Minesweeper>` in `AppState`.
- `GET /` renders the board; `POST /reveal`, `POST /flag`, `POST /new` mutate state and redirect back.
- The index handler builds `Vec<Vec<CellView>>` (which includes `prob_color` and `prob_pct` per cell) and passes it to `index.html` as `grid`.
- The template renders inline `background-color` CSS and `title` tooltip attributes from those fields.
- A small `<span class="prob-label">` shows the percentage inside each unopened cell; a JS snippet drives a status bar that updates on hover.

---

## Build & Run

```bash
# Build everything
cargo build

# Run individual front-ends
cargo run -p cli
cargo run -p gui
cargo run -p web   # serves http://127.0.0.1:8080
```

The workspace uses **resolver = "2"**. There are no integration tests yet; `cargo build` is the baseline check.

---

## Guidelines for Further Development

### Adding features to the core

- All game-rule changes belong in `minesweeper_core/src/lib.rs`. Front-ends must not implement game logic themselves.
- Public API surface: `reveal`, `toggle_flag`, `calculate_mine_probabilities`, and read access to `grid`, `state`, `mines_count`, `width`, `height`, `mines_generated`.
- Preserve the **lazy mine generation** invariant: mines must not be placed until the first `reveal` call, and the first-clicked cell must never be a mine.
- `CellContent` uses `#[serde(untagged)]` so `Empty(n)` serialises as the bare integer `n` and `Mine` serialises as the string `"Mine"`. The web template relies on this; do not change the representation without updating the template.

### Improving the probability estimator

- The current approach is **constraint-satisfied Monte Carlo**. It becomes less efficient (more attempts per valid sample) as more cells are revealed and constraints tighten.
- A natural next step is **constraint propagation**: identify cells that are *certainly* mines or *certainly* safe from the numbered constraints alone before running Monte Carlo on the remaining unknowns.
- Another improvement is **border-only sampling**: only the cells adjacent to at least one visible numbered cell are directly constrained; interior hidden cells can be handled analytically once the border is solved.
- When adding a smarter strategy, expose it through the same `calculate_mine_probabilities` signature so all front-ends benefit without changes.

### Adding a new front-end

1. Create a new crate under the workspace root and add it to `Cargo.toml`'s `members` list.
2. Depend only on `minesweeper_core = { path = "../minesweeper_core" }`.
3. Call `calculate_mine_probabilities` after each state-changing action and use the returned `Vec<Vec<f64>>` to drive your probability visualisation.

### Modifying an existing front-end

- **CLI**: the render loop redraws the entire screen on every iteration. Keep the draw path allocation-light; avoid recomputing probabilities inside the render loop (recompute only after game actions).
- **GUI**: `update_view` is synchronous and blocks the Qt event loop while computing probabilities. If the board grows or `MAX_VALID`/`MAX_ATTEMPTS` are increased significantly, move the computation to a background thread and emit `boardChanged` when done.
- **Web**: the template path is resolved at runtime relative to the working directory (`web/templates/**/*`). When running via `cargo run -p web`, the working directory must be the workspace root. The `Tera` instance is created once at startup and is not reloaded; restart the server after template changes during development.

### Code style

- Keep each crate focused: no game logic outside `minesweeper_core`, no rendering logic inside it.
- Prefer small, pure functions over large `match` trees.
- New public items in `minesweeper_core` should have doc comments.
- No `unwrap` in production paths of the web crate; propagate errors with `?` or return appropriate HTTP responses.
