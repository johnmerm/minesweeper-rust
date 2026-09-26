# The neural estimator: what it is and how it was trained

*Companion notebook: [`neural-estimator.ipynb`](neural-estimator.ipynb) — loads the
trained weights, runs them, and measures the model against the exact solver.*

A small convolutional network that predicts P(mine) for **one cell** from a 9×9
window around it. It is an approximation of a solver that is already exact, so
the point is not accuracy — it is that inference costs the same regardless of how
tangled the board is, where the exact search's cost is not bounded by anything so
simple.

It is also, in the web front-end, a thing to look at: the network's guess sits
beside the solver's proof, and where they differ is the interesting part.

---

## Architecture

```
input   (8, 9, 9)          one 9x9 patch, 8 channels
        │
        ├─ Conv 3x3 →32  + BatchNorm + ReLU
        ├─ Conv 3x3 →64  + BatchNorm + ReLU
        ├─ Conv 3x3 →64  + BatchNorm + ReLU
        ├─ Conv 3x3 →32  + BatchNorm + ReLU
        │
        ├─ AdaptiveAvgPool → (32, 3, 3) → flatten 288
        ├─ concat the global mines ratio                 → 289
        │
        ├─ Linear 289→128 + ReLU + Dropout(0.2)
        ├─ Linear 128→64  + ReLU
        └─ Linear  64→1   + Sigmoid                      → P(mine)
```

**122 049 trainable parameters**, or **121 665** in the shipped file once
BatchNorm is folded away. Source: `neural/model.py`.

Two details that are not decoration:

**The global ratio is appended after the pooling.** Mines-remaining ÷ cells-still-
hidden arrives as channel 6, constant across the patch — and average pooling over
a constant is that constant, so it would survive, but only as one value diluted
among 288. Concatenating it back gives the head an undiluted copy. It is the only
thing in the input that is not local, and on a board where no number speaks it is
the *only* signal there is.

**The receptive field is the patch.** Four 3×3 convolutions with padding reach
nine cells across, which is exactly the patch. So the network can in principle see
every number that constrains the target cell — a mine constrains only its eight
neighbours, and those neighbours' numbers are all within four cells — and nothing
whatever about the rest of the board except channel 6.

### Input channels

| Ch | Value | Meaning |
|----|-------|---------|
| 0 | number ÷ 8 | what a visible cell shows |
| 1 | {0,1} | is visible |
| 2 | {0,1} | is hidden |
| 3 | {0,1} | is flagged |
| 4 | {0,1} | off the edge of the board |
| 5 | 1 at (4,4) | *this* is the cell being asked about |
| 6 | ratio | mines remaining ÷ cells still hidden, broadcast |
| 7 | {0,1} | unopened and touching a visible number |

Two traps in there, both of which I got wrong first time writing the Python twin
for the notebook, and neither of which raises an error — they just shift an input
channel and the model returns confident nonsense:

- **Channel 6's denominator counts only `Hidden` cells, not flagged ones**, and its
  numerator subtracts mines *actually uncovered*, not flags the player has placed.
  (The solver treats a flag as an unknown; the ratio treats it as spoken for.
  Different questions.)
- **Channel 7 needs a visible neighbour showing a number greater than zero.** A
  visible `0` does not make a cell a border cell, because it constrains nothing.

The layout is written down in three places — `probability/patch.rs`,
`neural/dataset.py`, and now `notes/minesweeper_ref.py` — with nothing tying them
together. `notes/rust_reference.txt` is the tie: the notebook checks its patches
against probabilities the Rust engine produced for a fixed position.

---

## Training data

`datagen` (Rust) plays real games and labels them with **the exact constraint
solver** — the same code described in the other note. Configurations are sampled
uniformly from 9×9/10, 10×10/15, 16×16/40, 16×16/51 and 30×16/99, and each game is
played a random number of moves (`1..=w·h/4`) before being labelled, so positions
come from every stage of a game rather than only the end.

Two deliberate choices:

- **A position the solver cannot finish within budget is skipped, not sampled.**
  In the output a sampled label is indistinguishable from an exact one, so letting
  one through would quietly teach the model somebody's guesses.
- **Labels are probabilities, not mine/no-mine.** It is regression onto the
  solver's answer, not classification of the board. A cell the solver puts at 0.4
  should be predicted 0.4, not rounded to "safe".

**60 000 positions → 6 485 906 training cells**, 757 394 validation, ~380 000 test.

---

## The run

| | |
|---|---|
| Loss | `BCE + 0.1 · MSE` |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4, cosine annealing |
| Batch | 2048 |
| Augmentation | D4 — 4 rotations × 2 flips; the labels are symmetry-invariant |
| Epochs | 10, on CPU |

BCE gives calibration — it punishes a confident wrong answer far harder than a
hedged one, which for a probability is exactly right. The MSE term is small and
only discourages being wrong *by a lot*.

```
Epoch   1/10  train_loss=0.4868  val_bce=0.4906  val_mae=0.0800
Epoch   2/10  train_loss=0.4586  val_bce=0.4474  val_mae=0.0477
Epoch   3/10  train_loss=0.4414  val_bce=0.4324  val_mae=0.0356
Epoch   4/10  train_loss=0.4328  val_bce=0.4274  val_mae=0.0293
Epoch   5/10  train_loss=0.4300  val_bce=0.4260  val_mae=0.0296
Epoch   6/10  train_loss=0.4280  val_bce=0.4257  val_mae=0.0348
Epoch   7/10  train_loss=0.4269  val_bce=0.4241  val_mae=0.0300
Epoch   8/10  train_loss=0.4261  val_bce=0.4243  val_mae=0.0334
Epoch   9/10  train_loss=0.4256  val_bce=0.4229  val_mae=0.0274   <- best
Epoch  10/10  train_loss=0.4253  val_bce=0.4231  val_mae=0.0298
```

**This is a first run, not a tuned one.** The curve was still drifting down when
cosine annealing took the learning rate to zero. Ten epochs was what fit; it is
the most obvious thing to improve.

### Why it would not train before

Four separate reasons, none of them the model — recorded in `neural/README.md`,
summarised here because they are the interesting part:

1. **Memory.** The dataset parsed JSONL and kept every record as Python dicts, about
   22 KB each: 500 000 positions needed ~11.5 GB resident *before* `DataLoader`
   forked it per worker. The same record is 768 bytes as three uint8 planes, which
   is what `prepare.py` now writes and the dataset memory-maps — ~380 MB on disk,
   near-zero resident. Extraction went from 170 µs a sample to 18 µs as well.
2. **The data was all endgames.** `datagen` used to play every game down to ≤80
   hidden cells before labelling, and skipped Expert entirely, because the solver
   could not label anything larger in reasonable time. It can now — single-digit
   milliseconds — so the median position has ~255 hidden cells instead of under 80.
   A model trained on the old data had never seen a board that looked like the
   start of a game.
3. **The Rust side did not compile.** A transitive dependency of `tract-onnx` had
   moved past the project's toolchain, and `Cargo.lock` was untracked so every
   clone re-resolved to it.
4. **`export.py` died on a missing dependency** — `torch.onnx.export` needs
   `onnxscript` on current torch, and it was not in `requirements.txt`. It failed
   at the last step, after training had already been paid for.

---

## How good is it

On the held-out split, ~380 000 cells:

| | |
|---|---|
| mean absolute error | 0.027 |
| median | 0.016 |
| **proven mines it calls under 10%** | **0.143%** |
| proven-safe cells it calls over 50% | 1.06% |

Calibration is close to honest: of the cells it scores above 0.95, 99.7% really
are mines.

That 0.143% is the number that matters, because those are the ones that lose
games — 24 cells out of 16 771 where the solver *knew* there was a mine and the
model would have opened it. It is why the exact solver stays authoritative
everywhere, and why the network's guess never feeds auto-reveal.

The notebook measures this live and finds, as you would expect, that it depends
sharply on density: on beginner boards the model puts ~90% of proven-safe cells
under 5%, on intermediate ~85%, and on Expert mid-game positions the figure I
measured while building the web front-end was 76.5%. It is a local approximation,
and denser boards need more than local information.

---

## Two implementations, on purpose

`model.onnx` is read by `tract` in the desktop build. `model.bin` is the same
weights, flattened with BatchNorm folded into the preceding convolution, read by
`probability::patch_cnn` — the forward pass written out by hand in Rust.

The hand-written one exists for two reasons. **Size**: `tract` costs 13.4 MB in
the WebAssembly build against a 174 KB baseline. **And the gradient**: owning the
arithmetic means owning the derivative, which is what lets the network be
corrected during play.

Three implementations therefore have to agree — PyTorch, `patchcnn_reference.py`
in numpy, and the Rust — so `export_weights.py` writes `model.vectors`: patches
with the outputs PyTorch gave for them, letting the Rust test check itself with no
Python present.

### Folding BatchNorm

At inference BatchNorm is an affine map with fixed parameters, so it folds into
the convolution feeding it:

```
scale = gamma / sqrt(var + eps)
W'    = W * scale
b'    = (b - mean) * scale + beta
```

The Rust then sees convolution, bias, ReLU, and nothing else. `export_weights.py`
checks the folding end to end rather than trusting the algebra, and asserts it
changed the model by less than 1e-5.

---

## Correction during play

Every exact solve is a perfectly labelled position that cost nothing extra to
produce, so the web front-end uses it: after each solve the network's **output
layer** is stepped towards the solver's answers.

Only the last layer. With a sigmoid output and cross-entropy loss the gradient
into the head's weights is just `(prediction − target) × feature` — no chain rule
left to apply — which is a dozen lines of Rust rather than an autograd engine in
the browser.

Two gates, both of which exist for a reason:

- **Only after an exact solve.** A sampled estimate carries noise, and a network
  taught from noise learns the noise.
- **Only in the display modes where the solver is visible.** Under *neural network
  only* the page runs uncorrected, because a network being corrected by the solver
  mid-run is not the thing that mode is there to measure.

The correction **rides the scoring pass** rather than running as a second one,
because `learn` and `predict` perform the same forward pass. Done separately over
a whole board it measured **4.2 seconds on 40×40, on every move**; folded in, the
worst frame it costs is about 15 ms.

---

## Checking the patch against the engine

`notes/rust_reference.txt` holds the Rust engine's probabilities for one fixed
position. To regenerate it, drop this in `minesweeper_core/tests/`, run it, and
prepend the header:

```rust
use minesweeper_core::{CellContent, CellState, Minesweeper};

#[test]
fn dump() {
    let mines = [(0usize, 0usize), (2, 1), (3, 3), (5, 0), (4, 2)];
    let (width, height) = (6usize, 4usize);
    let mut game = Minesweeper::new(width, height, mines.len());
    for &(x, y) in &mines { game.grid[y][x].content = CellContent::Mine; }
    for y in 0..height { for x in 0..width {
        if matches!(game.grid[y][x].content, CellContent::Mine) { continue; }
        let mut count = 0u8;
        for dy in -1i32..=1 { for dx in -1i32..=1 {
            if dx == 0 && dy == 0 { continue; }
            let (nx, ny) = (x as i32 + dx, y as i32 + dy);
            if (0..width as i32).contains(&nx) && (0..height as i32).contains(&ny)
                && matches!(game.grid[ny as usize][nx as usize].content, CellContent::Mine) {
                count += 1;
            }
        }}
        game.grid[y][x].content = CellContent::Empty(count);
    }}
    game.mines_generated = true;
    for &(x, y) in &[(1usize, 1usize), (2, 2), (3, 1), (4, 1), (1, 2), (3, 2)] {
        game.grid[y][x].state = CellState::Visible;
    }
    game.grid[0][1].state = CellState::Flagged;

    let probs = minesweeper_core::probability::PatchCnn::from_bytes(
        &std::fs::read("../neural/onnx/model.bin").unwrap()).unwrap().calculate(&game);
    for y in 0..height { for x in 0..width {
        println!("{} {} {:.6}", x, y, probs[y][x]);
    }}
}
```

---

## Running the pipeline

```bash
cargo build --release -p datagen
python -m venv .venv && . .venv/bin/activate
pip install -r neural/requirements.txt

python neural/datagen.py 200000 8   # labelled positions -> data/{train,val,test}.jsonl
python neural/prepare.py            # -> data/{split}_{boards,cells,labels,ratios}.npy
python neural/train.py --epochs 30  # -> checkpoints/best.pt
python neural/eval.py               # how good is it, and where is it wrong?
python neural/export.py             # -> onnx/model.onnx  (desktop, via tract)
python neural/export_weights.py     # -> onnx/model.bin   (everywhere else)
./wasm/build.sh                     # model.bin is include_bytes!'d into the .wasm
```

`prepare.py` is not optional — the trainer reads the arrays it writes, not the
JSONL.

## Source

| What | Where |
|---|---|
| The model | `neural/model.py` |
| Patch layout | `minesweeper_core/src/probability/patch.rs`, `neural/dataset.py` |
| Data generation | `datagen/src/main.rs`, `neural/datagen.py` |
| Training | `neural/train.py`, `neural/prepare.py` |
| Export | `neural/export.py`, `neural/export_weights.py` |
| The numpy specification | `neural/patchcnn_reference.py` |
| Rust inference and the gradient step | `minesweeper_core/src/probability/patch_cnn.rs` |
| Tests | `minesweeper_core/tests/patch_cnn.rs`, `tests/neural.rs` |
