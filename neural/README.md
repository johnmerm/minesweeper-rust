# Neural mine-probability estimator

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
python neural/export.py             # -> onnx/model.onnx

cargo run -p gui --features neural   # the Rust side loads the ONNX file
```

`prepare.py` is not optional, and is the step that was missing.

## Why it would not train before

Two things, both outside the model:

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

## Files

| File | Purpose |
|------|---------|
| `datagen.py` | Runs the Rust `datagen` binary in parallel, splits train/val/test |
| `prepare.py` | JSONL -> compact arrays the trainer streams |
| `dataset.py` | Patch extraction and D4 augmentation |
| `model.py` | PatchCNN, ~180k parameters |
| `train.py` | Training loop, `--limit` for a quick smoke run |
| `export.py` | Checkpoint -> ONNX, validated against onnxruntime |
