# WebAssembly front-end

The Rust engine (`minesweeper_core`) compiled to `wasm32-unknown-unknown` plus a
static HTML/JS view. There is no server, no bundler and no `wasm-bindgen`: the
module exports a small C ABI (see `wasm/src/lib.rs`) and `minesweeper.js` is the
whole glue layer.

| File | Purpose |
|------|---------|
| `index.html` | The page — markup and styles |
| `minesweeper.js` | Loads the module, renders the board, handles input |
| `minesweeper.wasm` | Built artifact, committed so the site needs no build step |
| `model.bin` | The trained network's weights, copied here by the build |
| `standalone.html` | Generated: the above four inlined into one file |

`index.html` carries a build id — a hash of the built files — which
is shown at the bottom of the page and appended to both asset URLs. These files
are served from a CDN, so without it a browser can quietly keep an old
`minesweeper.js` beside a new `minesweeper.wasm`; with it, a page and the code it
loads always match, and the stamp answers "am I running the new build?" at a
glance.

## Playing it

Served from any static host. Straight from GitHub via **raw.githack.com**:

```
https://raw.githack.com/johnmerm/minesweeper-rust/main/docs/index.html
```

Swap `main` for a branch or tag name to open that version, and use
`rawcdn.githack.com` instead for the cached, rate-limit-free CDN copy.

githack caches, so a freshly pushed build may not appear at once: hard-reload
(Ctrl/Cmd-Shift-R), and check the build id at the foot of the page against the
one `./wasm/build.sh` printed.

`standalone.html` carries the module and the weights inlined as base64, so that
one file also runs from a `file://` URL — download it and double-click it. `index.html` needs
`http(s)` because browsers refuse to `fetch` a `.wasm` from `file://`.

Locally:

```bash
python3 -m http.server -d docs 8000   # then open http://localhost:8000/
```

## Rebuilding

```bash
rustup target add wasm32-unknown-unknown   # once
./wasm/build.sh
```

That compiles the `minesweeper_wasm` crate in release mode, copies the `.wasm`
into this directory and regenerates `standalone.html`. Commit the regenerated
artifacts — the site is served directly from the repository, so a stale
`minesweeper.wasm` means stale gameplay.

`standalone.html` is generated; edit `index.html` and `minesweeper.js` instead.

To check the artifact before committing it:

```bash
node wasm/smoke.mjs
```

It drives the same ABI the page uses — no npm install needed.

## How it plays

- Left click reveals, right click flags. *Flag mode* makes left click flag, for
  touch screens.
- Every hidden cell is tinted from grey to red by its estimated chance of
  hiding a mine, with the percentage in the corner.
- **Estimator** picks how those numbers are produced: exact constraint search,
  Monte Carlo sampling, or *auto*, which uses the exact search and falls back to
  sampling only when the search finds no valid layout. The line under the board
  reports how much work each one did; the highlighted line is the one being
  displayed.
- **Auto-play deductions** opens every cell proven safe (0%) and flags every cell
  proven to be a mine (100%), repeating until nothing further follows. It only
  acts on proof — never on a sampled estimate, where 0% merely means no sample
  happened to put a mine there.
- **Neural overlay** puts a second number in the top-left of each cell, in
  purple: what the trained convolutional network thinks, beside what the solver
  proved. The network sees only a 9x9 patch around a cell, so it is a guess and
  never a proof — it is there to be compared with the exact value, not trusted
  instead of it.

  Scoring a cell is a full forward pass and a large board has thousands, so the
  work is spread over animation frames: cells adjacent to a number are scored
  first, and the rest fill in over the next few seconds while the page stays
  responsive. A cell not yet reached simply shows nothing.

  Whenever the exact solver finishes a position, its answers are used to correct
  the network's output layer — a perfectly labelled example that cost nothing
  extra to produce. The note under the board reports how far off it was. Only
  exact solves teach it; a sampled estimate carries noise, and a network taught
  from noise learns the noise.

  The weights are `model.bin`, produced by `neural/export_weights.py`. Without
  them the button reports the overlay as unavailable and nothing else changes.
- Boards go up to 200 a side. That ceiling is about the browser, not the engine:
  every cell is a DOM node, and 200x200 is 40 000 of them.
