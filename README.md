# Minesweeper

A Rust minesweeper engine with four independent front-ends, and two estimators
that answer the only interesting question on the board: *what are the odds this
square is a mine?*

## Play it

Nothing to install — the engine is compiled to WebAssembly and the site is
served straight from this repository.

### ▸ **[Play in your browser](https://raw.githack.com/johnmerm/minesweeper-rust/main/docs/index.html)**

Or [download `standalone.html`](https://raw.githubusercontent.com/johnmerm/minesweeper-rust/main/docs/standalone.html)
— one file with the module, the script and the trained network all inlined, which
runs from a `file://` URL with no server and no network at all.

> githack caches. If a change seems missing, hard-reload (Ctrl/Cmd-Shift-R) and
> check the build id at the foot of the page. For a cached, rate-limit-free copy
> swap the host for `rawcdn.githack.com`.

## How it works

Both estimators are written up in prose, and again as notebooks that *run* what
they describe — every number in them is computed, not quoted.

| | |
|---|---|
| **[The exact solver](https://raw.githack.com/johnmerm/minesweeper-rust/main/docs/notes/constraint-estimator.html)** · [notebook](https://raw.githack.com/johnmerm/minesweeper-rust/main/docs/notes/constraint-estimator-notebook.html) | Constraints, propagation, independent regions, and the two traps that made it silently wrong |
| **[The neural estimator](https://raw.githack.com/johnmerm/minesweeper-rust/main/docs/notes/neural-estimator.html)** · [notebook](https://raw.githack.com/johnmerm/minesweeper-rust/main/docs/notes/neural-estimator-notebook.html) | A 122k-parameter CNN: architecture, training, how good it is, and the correction it gets during play |

[Both, with sources](https://raw.githack.com/johnmerm/minesweeper-rust/main/docs/notes/index.html) ·
[`notes/` in the repository](notes/)

### The short version

**The probabilities are exact.** For each unopened cell the solver reports the
fraction of consistent mine layouts that put a mine there — it counts them, it
does not sample them. A Monte Carlo estimator used to fill in when the exact
search gave up; it was deleted, because over 71 mid-game positions it answered 20
of them, took 4.4 seconds against the exact search's 20 milliseconds, and among
its answers was a cell it called 0% that was not safe.

What makes counting tractable is splitting the border into groups that share no
constraint, solving each separately, and convolving the results — a product
turned into a sum. That took the worst measured move from 137 seconds to
milliseconds.

**When it cannot be exact, it says so.** `calculate_mine_probabilities` returns
`Option`, and the page draws a `?` on a colour of its own rather than a number.
That matters more than it sounds: the grey of a 0% cell means *certainly safe*,
so an unsolved board drawn normally would claim the opposite of the truth. It
happens on roughly one move in twenty on a 30×30 board with 250 mines, and not at
all on the standard ones.

**The network is a comparison, never an authority.** It predicts one cell's
probability from a 9×9 patch around it and is right most of the time — but it
calls 0.14% of proven mines less than 10%, which is exactly the kind of wrong
that loses games. Its guess never drives auto-play unless you explicitly ask it
to, which the page lets you do so you can watch how it fares.

## The repository

```
minesweeper_core/   the engine and both estimators — all game logic lives here
├── probability/
│   ├── setup.rs            board → constraints, and what propagation settles
│   ├── components.rs       independent regions, and recombining them
│   ├── constraint_search.rs the exact count, with a node budget
│   └── patch_cnn.rs        the CNN forward pass and its gradient, by hand
cli/     terminal front-end (crossterm)
gui/     desktop front-end (Qt via qmetaobject)
web/     server-rendered front-end (Actix-Web + Tera)
wasm/    the C ABI the browser build talks to — no wasm-bindgen, no bundler
docs/    the static site, served from the repository  → docs/README.md
neural/  the training pipeline: datagen, prepare, train, export  → neural/README.md
notes/   the write-ups and their notebooks  → notes/README.md
datagen/ labelled positions for training, from the exact solver
```

`AGENTS.md` is the guide for changing any of it, including the parts that look
like details and are not.

## Build and run

```bash
cargo build                  # everything (gui needs Qt 5 development packages)

cargo run -p cli             # terminal
cargo run -p gui             # desktop
cargo run -p web             # http://127.0.0.1:8080

rustup target add wasm32-unknown-unknown   # once
./wasm/build.sh                            # rebuild docs/ from source
python3 -m http.server -d docs 8000        # then open http://localhost:8000/
```

`cargo build` is the baseline check. `node wasm/smoke.mjs` drives the built
`docs/minesweeper.wasm` through the same ABI the page uses and needs no npm
install; run it after `./wasm/build.sh`. `cargo test -p minesweeper_core` checks
the estimator against a brute-force oracle, among other things.

Commit the regenerated `docs/` artifacts whenever the core or the wasm crate
changes — the site is served straight from the repository, so a stale `.wasm`
ships stale gameplay.
