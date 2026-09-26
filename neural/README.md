# Neural mine-probability estimator

*In depth, with a runnable notebook: [`../notes/neural-estimator.md`](../notes/neural-estimator.md).*

A small CNN that predicts P(mine) for one cell from a 9x9 patch around it,
trained on exact labels from the constraint solver. It is an approximation of a
solver that is already exact — the point is speed, not accuracy: inference is a
single batched forward pass regardless of how tangled the board is.

## Running it

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

cargo run -p gui --features neural   # the Rust side loads the ONNX file
```

`prepare.py` is not optional, and is the step that was missing.

Verified end to end on this branch: 60 000 positions generated, prepared,
trained, exported, and loaded back by the Rust `neural` feature, which returns
per-cell probabilities that vary across the board — see
`minesweeper_core/tests/neural.rs`, which is skipped unless `MINESWEEPER_ONNX`
points at a model.

## Why it would not train before

Four things, none of them the model:

**Memory.** The dataset read JSONL and kept every parsed record — about 22 KB of
Python dicts each. The suggested 500 000 positions therefore needed ~11.5 GB
resident before `DataLoader` forked it once per worker, so the run died before
the first epoch. The same record is 768 bytes as three uint8 planes, which is
what `prepare.py` writes and what the dataset now memory-maps: 500 000 positions
cost about 380 MB on disk and almost nothing resident.

It was slow as well as large — the patch extractor recomputed the whole board and
its border mask for every cell, 64 times per record, in Python. Extraction is now
a slice of preprocessed planes: 18 us per sample against 170 us.

**The training data was all endgames.** `datagen` played every game down to 80 or
fewer hidden cells before labelling, and left Expert out entirely, because the
exact solver could not label anything larger in reasonable time. It can now — a
position labels in single-digit milliseconds — so games are sampled at every
stage, Expert is included, and the median position has ~255 hidden cells rather
than under 80. A model trained on the old data had never seen a board that looked
like the start of a game.

Positions the solver cannot finish within its budget are skipped rather than
labelled by sampling: in the output a sampled label is indistinguishable from an
exact one, so letting one through would quietly teach the model someone's guesses.

**The Rust side did not compile.** `cargo build --features neural` failed because
a transitive dependency of tract-onnx had moved past the project's toolchain, and
`Cargo.lock` is untracked so every clone re-resolved to it. The workspace now
declares its `rust-version` and uses resolver 3, which picks versions that build.

**And `export.py` died on a missing dependency** — `torch.onnx.export` needs
`onnxscript` on current torch, which was not in requirements.txt. It failed at the
last step, after training had already been paid for.

## Files

| File | Purpose |
|------|---------|
| `datagen.py` | Runs the Rust `datagen` binary in parallel, splits train/val/test |
| `prepare.py` | JSONL -> compact arrays the trainer streams |
| `dataset.py` | Patch extraction and D4 augmentation |
| `model.py` | PatchCNN, ~180k parameters |
| `train.py` | Training loop, `--limit` for a quick smoke run |
| `export.py` | Checkpoint -> ONNX, validated against onnxruntime |
| `export_weights.py` | Checkpoint -> flat weights the Rust build reads without tract |
| `patchcnn_reference.py` | The same forward pass in plain numpy, as the specification |
| `eval.py` | Scores a checkpoint against the exact labels on the held-out split |
| `check.py` | Verifies the prepared arrays against the JSONL, no torch needed |

## The trained model

`onnx/` holds a model trained on this branch: 60 000 positions, 6.5M cells,
10 epochs. On the held-out split, over 380 000 cells:

| | |
|---|---|
| mean absolute error | 0.027 |
| median | 0.016 |
| proven mines it calls under 10% | **0.143%** — the ones that lose games |
| proven-safe cells it calls over 50% | 1.06% — only wasted moves |

Calibration is close to honest: of the cells it scores above 0.95, 99.7% really
are mines; of those it scores 0.30-0.50, 40% are.

That 0.143% is why the exact solver stays authoritative. It is 24 cells out of
16 771 where the solver *knew* there was a mine and the model would have opened
it. A good approximation is still an approximation, and there is no reason to act
on one when the exact answer takes twelve milliseconds.

## Two implementations, on purpose

`model.onnx` is read by tract in the desktop build. `model.bin` is the same
weights, flattened with BatchNorm folded in, for `probability::patch_cnn` — the
network written out by hand.

The hand-written one exists because tract costs 13.4 MB in the WebAssembly build
against a 174 KB baseline, and because owning the arithmetic means owning the
gradient, which is what lets the network be corrected during play from the
solver's exact answers.

Being able to disagree is the risk, so three things are checked against each
other: PyTorch, `patchcnn_reference.py`, and the Rust. `export_weights.py` writes
`model.vectors` — patches with the outputs PyTorch gave — so the Rust test can
verify itself with no Python present.
