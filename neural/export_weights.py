#!/usr/bin/env python3
"""Export PatchCNN's weights as a flat binary the Rust side reads directly.

Usage:
    python neural/export_weights.py [--checkpoint checkpoints/best.pt]
                                    [--output onnx/model.bin]

# Why not just ship the ONNX

Because of what it costs to read one. Running the ONNX model needs `tract`, and
compiling `tract` into the WebAssembly build takes the module from 174 KB to
13.4 MB — a seventy-sevenfold increase, on a page whose whole point is that it
loads instantly from a CDN as three static files.

The architecture is fixed and small: four convolutions and three linear layers,
122k parameters. Writing that forward pass out by hand costs no dependency and
essentially no code size, and it also hands us the gradient needed to adjust the
last layer during play. The ONNX export stays for the desktop build, where size
does not matter.

# Folding BatchNorm

Each convolution is followed by BatchNorm, which at inference is an affine map
with fixed parameters. Folding it into the preceding convolution removes it
entirely:

    scale = gamma / sqrt(var + eps)
    W'    = W * scale
    b'    = (b - mean) * scale + beta

So the Rust side sees convolution, bias, ReLU, and nothing else.

# Layout

Magic `MSPCNN1\\0`, then little-endian f32 arrays back to back, in this order:

    conv0 W [32,8,3,3]   b [32]      conv1 W [64,32,3,3]  b [64]
    conv2 W [64,64,3,3]  b [64]      conv3 W [32,64,3,3]  b [32]
    fc0   W [128,289]    b [128]     fc1   W [64,128]     b [64]
    fc2   W [1,64]       b [1]

Shapes are implied, not stored, so the Rust reader checks the total length. Any
change to the architecture must change the magic.
"""

import argparse
import struct
from pathlib import Path

import torch

from model import PatchCNN

MAGIC = b"MSPCNN1\0"
CKPT_DEFAULT = Path(__file__).parent / "checkpoints" / "best.pt"
OUT_DEFAULT = Path(__file__).parent / "onnx" / "model.bin"


def fold_batchnorm(conv_w, conv_b, gamma, beta, mean, var, eps=1e-5):
    """Absorb a BatchNorm into the convolution that feeds it."""
    scale = gamma / torch.sqrt(var + eps)
    folded_w = conv_w * scale.reshape(-1, 1, 1, 1)
    folded_b = (conv_b - mean) * scale + beta
    return folded_w, folded_b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CKPT_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"no checkpoint at {args.checkpoint} — train first")

    model = PatchCNN()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.eval()
    state = model.state_dict()

    tensors = []
    # conv index -> the BatchNorm that follows it, as laid out in model.py
    for conv, norm in ((0, 1), (3, 4), (6, 7), (9, 10)):
        weight, bias = fold_batchnorm(
            state[f"conv.{conv}.weight"],
            state[f"conv.{conv}.bias"],
            state[f"conv.{norm}.weight"],
            state[f"conv.{norm}.bias"],
            state[f"conv.{norm}.running_mean"],
            state[f"conv.{norm}.running_var"],
        )
        tensors += [weight, bias]

    for layer in (0, 3, 5):
        tensors += [state[f"fc.{layer}.weight"], state[f"fc.{layer}.bias"]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as handle:
        handle.write(MAGIC)
        for tensor in tensors:
            flat = tensor.detach().flatten().to(torch.float32).numpy()
            handle.write(struct.pack(f"<{flat.size}f", *flat))

    values = sum(t.numel() for t in tensors)
    print(f"Wrote {args.output} — {values:,} values, {args.output.stat().st_size:,} bytes")

    # The folding is only correct if it changes nothing, so check it end to end
    # rather than trusting the algebra.
    import numpy as np

    from patchcnn_reference import forward as reference_forward

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(16):
        patch = rng.random((8, 9, 9), dtype=np.float32)
        with torch.no_grad():
            expected = float(model(torch.from_numpy(patch).unsqueeze(0))[0])
        actual = reference_forward(args.output, patch)
        worst = max(worst, abs(expected - actual))
    print(f"Folded weights reproduce the model to {worst:.2e}")
    assert worst < 1e-5, f"folding changed the model by {worst}"

    # Test vectors, so the Rust implementation can check itself against PyTorch
    # without PyTorch being present. Without these the two implementations of this
    # architecture are only ever compared by hand, which is to say eventually not
    # at all.
    vectors = args.output.with_suffix(".vectors")
    patches = []
    for _ in range(24):
        patch = np.zeros((8, 9, 9), dtype=np.float32)
        for row in range(9):
            for col in range(9):
                roll = rng.random()
                if roll < 0.35:
                    patch[1, row, col] = 1.0
                    patch[0, row, col] = rng.integers(0, 9) / 8.0
                elif roll < 0.85:
                    patch[2, row, col] = 1.0
                elif roll < 0.92:
                    patch[3, row, col] = 1.0
                else:
                    patch[4, row, col] = 1.0
                patch[7, row, col] = float(rng.random() < 0.4)
        patch[5, 4, 4] = 1.0
        patch[6, :, :] = rng.random() * 0.3
        patches.append(patch)

    with torch.no_grad():
        expected = model(torch.from_numpy(np.stack(patches))).numpy()

    with open(vectors, "wb") as handle:
        handle.write(b"MSPCNNV1")
        handle.write(struct.pack("<I", len(patches)))
        for patch, value in zip(patches, expected):
            flat = patch.reshape(-1)
            handle.write(struct.pack(f"<{flat.size}f", *flat))
            handle.write(struct.pack("<f", float(value)))
    print(f"Wrote {vectors} — {len(patches)} patches with their PyTorch outputs")


if __name__ == "__main__":
    main()
