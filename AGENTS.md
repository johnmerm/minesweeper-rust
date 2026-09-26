# Minesweeper – Project Overview & Development Guidelines

## Project Structure

This is a Rust workspace with a shared game engine and four independent front-ends.

```
minesweeper/
├── minesweeper_core/   # Game logic library (shared by all front-ends)
├── cli/                # Terminal UI (crossterm)
├── gui/                # Desktop GUI (Qt / qmetaobject QML)
├── web/                # Web server (Actix-Web + Tera templates)
├── wasm/               # WebAssembly bindings (C ABI, no wasm-bindgen)
├── docs/               # Static HTML/JS site that loads the .wasm
└── notes/              # Write-ups of the two estimators, prose + runnable notebooks
```

`notes/` explains both estimators in depth, with notebooks that run what they
describe against the committed weights and check themselves against a
brute-force oracle, the Rust engine and PyTorch's own outputs. Read those before
changing `probability/` or the training pipeline.

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
| `Minesweeper::calculate_mine_probabilities` | Exact probability estimator – see below |

#### Mine probability estimator

`calculate_mine_probabilities(&self) -> Vec<Vec<f64>>`

Returns, for every unopened cell, the fraction of consistent mine layouts in which
it holds a mine. Exact, not sampled:

1. `SimSetup::build` collects the unopened cells and turns every visible number
   into a constraint over its unopened neighbours, then runs `propagate` to settle
   whatever local rules alone can settle.
2. `components::decompose` splits what remains into groups that share no
   constraint. They are independent apart from the board's total mine count, so
   solving them separately turns a product of their solution counts into a sum.
3. Each group is enumerated depth-first, recording `ways[k]` — layouts using
   exactly `k` mines — and `cell_ways[c][k]`.
4. `components::combine` convolves those, folds in `C(interior, mines_left)` for
   the cells no number speaks about, and divides out.

`ConstraintSearch::max_nodes` bounds the search. Past it the strategy reports no
result and the caller falls back to `MonteCarlo`, which is bounded too and gives
up once it is clear no valid sample is coming. `ConstraintSearch::exhaustive()`
removes the bound for offline work.

The estimators are fast enough to be called on every move at any board size:
worst measured single solve is ~12 ms on boards up to 200 a side, against 137
seconds before decomposition.

**Never treat a probability of 0.0 or 1.0 as merely a small or large number.**
Every front-end reads 0.0 as proof that a cell is safe and opens it. The solver
decides those two values from integer layout counts, never from the computed
ratio, precisely so that a weight underflowing in the tails cannot manufacture a
proof. A sampled 0% means only that no draw happened to put a mine there.

---

### Neural estimator (`neural` feature)

An optional CNN that predicts one cell's probability from a 9x9 patch, trained on
exact labels from the solver. Off by default; `cargo build --features neural`.
See `neural/README.md` for the pipeline and for the four separate reasons it could
not be trained before.

The patch layout is written down twice — `probability/neural.rs` and
`neural/dataset.py` — and nothing ties them together. Change one and change the
other, or the model is served inputs it never saw and returns confident nonsense
rather than an error. `minesweeper_core/tests/neural.rs` catches the drift.

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

#### Independent regions

`probability::components` splits the constraint graph into groups of cells that
share no constraint, solves each on its own, and recombines them. Enumerating the
border as one problem walks the *Cartesian product* of those groups' solutions;
solving them separately turns that into a sum, which is what makes a large board
tractable at all.

A group's result is deliberately not a probability but `ways[k]` (layouts using
exactly `k` mines) and `cell_ways[c][k]`. That form is independent of the global
mine count and of every other group, which is what lets them be combined — and
would let them be cached across moves. `combine` does the convolution, folding in
`C(interior, mines_left)` for the cells no number speaks about.

Two things to preserve when touching this:

- The scale factors in `scaled_binomials` cancel only because the same table
  divides numerator and denominator. Never compare weights across two tables.
- If any component exhausts its node budget the whole answer is discarded. A
  partial depth-first walk has covered a lexicographic prefix, so a cell can read
  0% purely because its subtree was never visited — and every caller reads 0% as
  proof that a cell is safe to open.

### `wasm`

WebAssembly front-end, built for `wasm32-unknown-unknown` with **no wasm-bindgen
and no bundler** so the result can be served as static files from any host
(GitHub Pages, raw.githack.com, or a `file://` URL).

- `wasm/src/lib.rs` exports a hand-rolled C ABI: commands (`ms_new`, `ms_reveal`,
  `ms_flag`, `ms_compute`, `ms_auto_reveal`) plus pointers to three flat buffers
  (`ms_cells_ptr` bytes, `ms_probs_ptr` `f32`s, `ms_stats_ptr` `u32`s). The module
  has **zero imports** — keep it that way, it is what makes the plain
  `WebAssembly.instantiate` load possible.
- Buffers are reallocated by `ms_new` and linear memory can grow on any call, so
  the JS side must re-read every pointer and rebuild its typed-array views after
  each call. `docs/minesweeper.js` does this in `cells()` / `probs()` / `stats()`.
- `wasm/src/rng.rs` registers a custom `getrandom` backend seeded from JS through
  `ms_seed`. `getrandom`'s usual wasm backend is generated by wasm-bindgen; this
  replaces it. `rand::thread_rng` seeds itself from it the first time it is used,
  so `ms_seed` must be called before the first `ms_reveal`.
- `docs/` is the site: `index.html` + `minesweeper.js` + the committed
  `minesweeper.wasm`, plus a generated `standalone.html` with everything inlined
  as base64 for `file://` use. Run `./wasm/build.sh` and commit the regenerated
  artifacts whenever the core or the wasm crate changes — the site is served
  straight from the repository, so a stale `.wasm` ships stale gameplay.
- Everything runs on the browser's main thread, so both ends are bounded:
  `ConstraintSearch` has a node budget (past it, it reports nothing and the caller
  falls back to sampling), and `scheduleCompute` in `minesweeper.js` defers the
  calculation past the repaint so a click never blocks on it.
- `render` skips cells whose appearance has not changed. On a 120×120 board
  repainting all 14 400 every move cost seconds — far more than the estimators.
- `wasm/bundle.py` stamps `index.html` with a hash of the built script and module,
  appends it to both asset URLs and shows it on the page. These files are served
  straight from a CDN, so without it a stale `minesweeper.js` can be paired with a
  fresh `.wasm`. The id is content-derived, so an unchanged rebuild is a no-op in
  the diff — never replace it with a timestamp or a commit hash.
- `MAX_DIM` caps boards at 200 a side. That is a guard on DOM size, not an engine
  limit.
- The neural overlay is the same architecture as `neural/`, written out by hand in
  `probability::patch_cnn` — `tract` costs 13 MB in wasm, against 30 KB for the
  forward pass. The weights are `include_bytes!`d from `neural/onnx/model.bin`, so
  the module carries them and there is no second asset a host can serve wrongly;
  `ms_model_load` parses them on first use. `ms_neural_begin` and
  `ms_neural_step(budget)` then score the board a few cells at a time so the page keeps its frames; `BoardScorer` puts the cells next to a number
  first. Unscored cells read `-1`, not 0 — nearly-zero is a real answer here.
  `ms_neural_begin(rate_millis)` corrects the output layer from `ms_probs_ptr` as
  it scores, because `learn_scored` and `predict` are the same forward pass —
  correcting separately over a whole board measured 4.2 s on 40x40 and blocked
  every move. `minesweeper.js` passes a non-zero rate only after an *exact* solve
  (a sampled estimate is noise, and the network would learn it) and only when the
  solver is on screen, so `Show: neural network only` runs uncorrected. A
  training pass records what the network said *before* each update, so the next
  pure scoring pass legitimately reads differently. `ms_neural_error` reports the
  mean gap. The guess never feeds `ms_auto_reveal`, which acts on proof alone.
- `ms_neural_auto(open_below, flag_above)` is the deliberate exception: auto-play
  driven by the network's estimates, per-mille thresholds instead of proof. It is
  a separate export from `ms_auto_reveal` precisely so the two can never be
  confused — this one opens mines, and is meant to, because that is how the model
  gets measured. It refuses to act unless *every* unopened cell has been scored:
  an unreached cell reads `NOT_SCORED`, which is below every threshold and would
  be opened as the safest cell on the board. The page reaches it only from the
  `Show: neural network only` mode, where `ms_auto_reveal` is not called at all.
- The `Show` select drives `showMode` in `minesweeper.js` ('both' / 'exact' /
  'neural'). In 'neural' nothing on the page reports the proved value — tint,
  label, tooltip and hover all switch over — since the mode exists to watch the
  network unaided. Scoring is skipped above `NEURAL_AUTO_MAX_CELLS` only while
  the mode is still the page's own default; choosing one is the asking.
  `ms_new` rebuilds `AppState` but carries the loaded network over, and
  `neural_probs` starts at `NOT_SCORED`, never `0.0` — it shipped once as `0.0`
  and every new board then showed a full grid of 0% from a network that had not
  looked at it.

---

## Build & Run

```bash
# Build everything
cargo build

# Run individual front-ends
cargo run -p cli
cargo run -p gui
cargo run -p web   # serves http://127.0.0.1:8080

# WebAssembly front-end: build the module and refresh docs/
rustup target add wasm32-unknown-unknown   # once
./wasm/build.sh
python3 -m http.server -d docs 8000        # then open http://localhost:8000/
```

The workspace uses **resolver = "2"**. `cargo build` is the baseline check. The
only automated test is `node wasm/smoke.mjs`, which exercises the built
`docs/minesweeper.wasm` through the same ABI the page uses (no npm install
required); run it after `./wasm/build.sh`.

---

## Guidelines for Further Development

### Adding features to the core

- All game-rule changes belong in `minesweeper_core/src/lib.rs`. Front-ends must not implement game logic themselves.
- `probability::certain_cells` returns what constraint propagation alone proves —
  mines and safe cells — without any search. It is the cheap way to make progress:
  auto-play iterates on it and pays for a full solve only once it runs dry. It is
  sound but incomplete, so a caller must never read "not proven" as "not certain".
- Public API surface: `reveal`, `toggle_flag`, `calculate_mine_probabilities`, and read access to `grid`, `state`, `mines_count`, `width`, `height`, `mines_generated`.
- Preserve the **lazy mine generation** invariant: mines must not be placed until the first `reveal` call, and the first-clicked cell must never be a mine.
- `CellContent` uses `#[serde(untagged)]` so `Empty(n)` serialises as the bare integer `n` and `Mine` serialises as the string `"Mine"`. The web template relies on this; do not change the representation without updating the template.

### Improving the probability estimator

The cheap wins are taken: constraint propagation, border-versus-interior
separation, independent-region decomposition, and reuse of region solutions
between moves all landed, and together took the worst case from 137 s to ~12 ms.
What is left is the case decomposition cannot help — a single region too large to
enumerate, where the search hits its budget and the answer falls back to sampling.

The next step there is a dynamic program over the search frontier: merge partial
assignments that agree on the cells still in play and on how many mines they have
used, instead of walking each one to a leaf. That is polynomial where the border
is thin, which it usually is. It would subsume decomposition rather than replace
it, since disconnected regions are just frontiers that never meet.

Whatever the strategy, keep the same two guarantees: bounded work, and no 0.0 or
1.0 that is not proven.

### Adding a new front-end

1. Create a new crate under the workspace root and add it to `Cargo.toml`'s `members` list.
2. Depend only on `minesweeper_core = { path = "../minesweeper_core" }`.
3. Call `calculate_mine_probabilities` after each state-changing action and use the returned `Vec<Vec<f64>>` to drive your probability visualisation.

### Modifying an existing front-end

- **CLI**: the render loop redraws the entire screen on every iteration. Keep the draw path allocation-light; avoid recomputing probabilities inside the render loop (recompute only after game actions).
- **Auto-reveal, in every front-end**: iterate on `probability::certain_cells`, and
  consult a full solve only once propagation has run dry. Re-solving after each
  pass is what made one click take 31 seconds. Only ever act on a *proven* 0% —
  the exact search's, never sampling's.
- **GUI**: `update_view` is synchronous and blocks the Qt event loop while computing probabilities. If the board grows or `MAX_VALID`/`MAX_ATTEMPTS` are increased significantly, move the computation to a background thread and emit `boardChanged` when done.
- **Web**: the template path is resolved at runtime relative to the working directory (`web/templates/**/*`). When running via `cargo run -p web`, the working directory must be the workspace root. The `Tera` instance is created once at startup and is not reloaded; restart the server after template changes during development.

### Code style

- Keep each crate focused: no game logic outside `minesweeper_core`, no rendering logic inside it.
- Prefer small, pure functions over large `match` trees.
- New public items in `minesweeper_core` should have doc comments.
- No `unwrap` in production paths of the web crate; propagate errors with `?` or return appropriate HTTP responses.
