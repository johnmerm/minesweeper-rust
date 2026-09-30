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

`notes/render.py` renders them to `docs/notes/*.html`, which is what a reader
gets from raw.githack — that host serves a `.md` as plain text, since Markdown is
a source format and nothing on that path renders it. `wasm/build.sh` runs it, so
a stale rendering cannot drift from its source the way a stale `.wasm` would.

It is stdlib only so a bare checkout can rebuild the site without a pip install,
and so one style covers the prose pages and the notebook pages alike —
`nbconvert` handles only the latter. Referencing a CDN would be fine; what
`standalone.html` guarantees is that the game needs no server, not that nothing
is ever fetched. The converter is a *partial* Markdown implementation: extend it
for a construct it lacks rather than assuming it works.

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
| `session::run` | The worker thread every threaded front-end uses: owns the `ConstraintSearch` and the `PatchCnn`, takes a `Job` per move, answers with `Reply`s |
| `session::Controller` | One game driven through that thread: which replies are stale, when the network is corrected and when it plays, the clock |

#### Mine probability estimator

`calculate_mine_probabilities(&self) -> Vec<Vec<f64>>`

Returns, for every unopened cell, the fraction of consistent mine layouts in which
it holds a mine. Exact, not sampled:

1. `setup::SimSetup::build` collects the unopened cells and turns every visible number
   into a constraint over its unopened neighbours, then runs `propagate` to settle
   whatever local rules alone can settle.
2. `components::decompose` splits what remains into groups that share no
   constraint. They are independent apart from the board's total mine count, so
   solving them separately turns a product of their solution counts into a sum.
3. Each group is enumerated depth-first, recording `ways[k]` — layouts using
   exactly `k` mines — and `cell_ways[c][k]`.
4. `components::combine` convolves those, folds in `C(interior, mines_left)` for
   the cells no number speaks about, and divides out.

`ConstraintSearch::max_nodes` bounds the search. Past it the strategy returns
`None` — no answer — and **nothing substitutes for it**: there is no sampled
fallback, by design. `ConstraintSearch::exhaustive()` removes the bound for
offline work.

Fast enough to run on every move at any board size: a solve is milliseconds on
the standard boards, and the worst measured at the current budget is ~375 ms on
the dense ones, against 137 seconds before decomposition. The page defers it
past the repaint, so a click never waits on it.

**Never treat a probability of 0.0 or 1.0 as merely a small or large number.**
Every front-end reads 0.0 as proof that a cell is safe and opens it. The solver
decides those two values from integer layout counts, never from the computed
ratio, precisely so that a weight underflowing in the tails cannot manufacture a
proof. There is no sampled estimator any more — see *Improving the probability
estimator* for the measurements that retired it.

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

Terminal front-end using **crossterm 0.27**, on the same `session::Controller`
as the Qt GUI and the web server, so a key never waits on a solve.

- Arrows or `hjkl` move; `Space`/`Enter` reveals; `F` flags; `N` new game;
  `1`/`2`/`3` the three presets; `A` auto-play; `S` cycles *Show*; `P` toggles
  labels; `Q` quits. `cli W H M` sets the opening board.
- The loop polls keys every 50 ms, drains the session's replies, and redraws only
  when something changed — overwriting lines in place, since the network reports
  several times a second and a full clear per report flickers.
- A terminal cell holds one label, so each square shows the estimate the tint
  follows, and the line under the board gives both for the cursor's cell.
- Boards larger than the terminal scroll to follow the cursor.

### `gui`

Desktop front-end using **qmetaobject 0.2** (Qt 5 bindings). It mirrors the page
in `docs/` — same controls, same colours, same wording — so a change to one
front-end's behaviour is a change to make in the other.

- `src/main.qml` is the UI, pulled in with `include_str!`; `src/main.rs` is the
  `MinesweeperGui` object behind it. Everything slow is `minesweeper_core::session`.
- **One `session::run` thread for the life of the window**, not one per move. It
  owns the `ConstraintSearch` (whose component cache only pays off if the
  instance survives between moves, exactly as in the wasm `AppState`) and the
  `PatchCnn`, which corrects itself as it scores. `session::Controller` sends a
  `Job` after every move; replies come back on the Qt thread through
  `queued_callback`, so nothing polls.
- Every job and reply carries a `generation`. A reply about an older board is
  dropped, and a newer job interrupts a scoring pass in progress.
- Cells are a `SimpleListModel<CellView>` in a `GridView`. `repaint` compares
  each cell against `painted` and calls `change_line` only for the ones that
  changed; the model is reset only when the board changes shape.
- The network is the core's `PatchCnn` with `neural/onnx/model.bin` compiled in,
  like the wasm build — not the `tract`/ONNX `NeuralNetwork`, so it needs no
  feature flag and no model file at run time.
- The page's rules carry over unchanged: an unsolved board shows `?` on the
  `UNKNOWN` tint, never a stale or zero tint; unscored cells hold `NOT_SCORED`;
  the network is corrected only after an exact solve and never in *Neural network
  only*; proof auto-play (`session::auto_reveal`) and the network's
  (`session::neural_auto`) are separate functions that are never confused.

### `web`

Web front-end using **Actix-Web 4** with **Tera** templates. One process serves
three things:

- **`/`** — the server-side game. One `session::Controller` behind a mutex, shared
  by every tab. The page (`templates/index.html`) is `docs/index.html`'s markup and
  CSS with `docs/minesweeper.js`'s drawing code, driven over HTTP instead of the
  wasm ABI: moves are `POST /reveal`, `/flag`, `/new`, `/settings` as JSON, each
  answering with the whole board, and while `busy` the page polls
  `GET /state?since=<revision>`, which answers `204` when nothing changed. Change
  either page and change the other. Cells use the wasm crate's byte encoding.
  An unsolved board sends `solved: false` and an empty `probs` — never zeros.
- **`/wasm/`** — `docs/` as a static site, read from disk per request so a
  `./wasm/build.sh` shows up on reload. `web --wasm` serves only that, on port
  8081, so the two can run side by side.
- **`/terminal`** — the `cli` binary itself, on a pseudo-terminal per tab, drawn by
  xterm.js over a WebSocket (`src/terminal.rs`). It runs only the `cli` built
  beside the server, never a shell; the browser chooses nothing but a board size,
  parsed as numbers; the handshake is refused unless `Origin` is this server,
  because a WebSocket is not covered by the same-origin policy; at most eight
  run at once; and a closed tab kills its process. Keep all of that if you touch it.

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
  `ConstraintSearch` has a node budget (past it, it reports nothing at all and
  the page shows `?`), and `scheduleCompute` in `minesweeper.js` defers the
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
cargo run -p web   # serves http://127.0.0.1:8080 (/, /wasm/, /terminal)
cargo run -p web -- --wasm   # only the WebAssembly site, on :8081
cargo build -p cli # /terminal runs this binary; build it first

# WebAssembly front-end: build the module and refresh docs/
rustup target add wasm32-unknown-unknown   # once
./wasm/build.sh
python3 -m http.server -d docs 8000        # then open http://localhost:8000/
```

The workspace uses **resolver = "3"** with `rust-version` pinned, so dependencies
resolve to versions the pinned toolchain can build — a crate that needs a newer
compiler is refused rather than picked. `minesweeper_core` is optimised even in
debug builds (`[profile.dev.package.minesweeper_core]`): unoptimised, the network
takes 128 s to score an Expert board, against 2.4 s. `cargo build` is the baseline check. The
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
enumerate, where the search hits its budget.

#### The requirement

**Every probability the estimator reports must be the true one.** Not only the
0.0 and the 1.0 — all of them. A number a player reads off a cell is a claim
about how the board actually is, and an approximation that looks the same as an
exact answer is a wrong claim dressed as a right one.

Monte Carlo used to fill the gap and has been **deleted**. It could not meet
that bar, and measured against the decomposing exact search it could not even
beat it on speed. Over 71 mid-game positions:

| | exact | Monte Carlo |
|---|---|---|
| answered | 71 of 71 | 20 of 71 |
| total time | 20 ms | 4 449 ms |
| worst error where both answered | — | 0.322 |
| cells called 0% that were not safe | — | 1 |

That last row is the one that settles it. A sampled 0% is read by every
front-end as proof, and on 30x16/99 it was wrong.

So the API says so now: `ProbabilityStrategy::calculate` and
`Minesweeper::calculate_mine_probabilities` return `Option<Vec<Vec<f64>>>`, and
`None` means *no answer*. Nothing substitutes for it. In the wasm ABI the same
fact is `STAT_SOLVED`, and the page draws a distinct colour and a `?` rather
than percentages — `probColor(0)` is the grey that means "certainly safe", so an
unsolved board left untinted would read as a board with no mines on it.

Exact counting of consistent layouts is #P-hard, so *exact always* and *fast
always* cannot both be promised. What is promised is that a number on screen is
correct, and that a position which cannot be solved says so.

#### What is still refused, and why the budget cannot fix it

Driving `docs/minesweeper.wasm` through 25 games per configuration, counting
every `ms_compute`:

| board | solves | unsolved | worst solve |
|---|---|---|---|
| 30x16/99 | 190 | 1 (0.5%) | 78 ms |
| 30x30/250 | 148 | 8 (5.4%) | 127 ms |
| 40x40/400 | 274 | 13 (4.7%) | 245 ms |

Raising `DEFAULT_MAX_NODES` is the obvious idea and it does not work. On
30x30/250 the *same 8* positions are refused at 1M, 4M, 8M and 32M nodes — the
budget buys nothing there and the worst solve grows from 127 ms to 2 935 ms,
because the search spends all of it before giving up. Those positions are not a
few times past an arbitrary line; they are genuinely large. Measure before
touching this constant: an earlier run on an easier trajectory suggested every
refusal was within reach, and it was not.

#### The next step

A dynamic program over the search frontier: merge partial assignments that agree
on the cells still in play and on how many mines they have used, instead of
walking each to a leaf. It is exact — it counts the same layouts, it just stops
re-deriving shared suffixes — and polynomial where the border is thin, which it
usually is. It would subsume decomposition rather than replace it, since
disconnected groups are just frontiers that never meet.

That, not a bigger budget, is what closes the remaining few percent.

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
- **GUI**: nothing slow runs on the Qt thread — send it to the worker as part of a
  `Job`. Emit `board_changed` only when the board's shape changes (QML resizes the
  window on it, as `onBoard_changed`: Qt capitalises the first letter, and
  `onboard_changed` silently never fires); per-move updates go through the cell
  model and `status_changed`.
- **Every threaded front-end** (Qt, web, CLI) goes through `session::Controller`;
  don't reimplement its bookkeeping in a front-end — that is how they drifted
  apart. Wording and drawing stay in each front-end. The wasm crate does the same
  work inline, because `wasm32-unknown-unknown` has no threads.
- **Web**: the template path is resolved at runtime relative to the working directory (`web/templates/**/*`). When running via `cargo run -p web`, the working directory must be the workspace root. The `Tera` instance is created once at startup and is not reloaded; restart the server after template changes during development.

### Code style

- Keep each crate focused: no game logic outside `minesweeper_core`, no rendering logic inside it.
- Prefer small, pure functions over large `match` trees.
- New public items in `minesweeper_core` should have doc comments.
- No `unwrap` in production paths of the web crate; propagate errors with `?` or return appropriate HTTP responses.
