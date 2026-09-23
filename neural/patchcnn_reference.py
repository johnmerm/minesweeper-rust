#!/usr/bin/env python3
"""A plain numpy reading of the exported weights file.

This is the specification the Rust implementation must match, written out in the
most obvious way possible. It exists so that three things can be checked against
each other rather than two: PyTorch, this, and Rust. If Rust and PyTorch disagree,
this says which of them drifted.

Deliberately unoptimised — loops where a library call would do — because its job
is to be transparently correct, not fast.
"""

import struct

import numpy as np

MAGIC = b"MSPCNN1\0"

# (out, in, kernel) per convolution, then (out, in) per linear layer.
CONVS = [(32, 8), (64, 32), (64, 64), (32, 64)]
LINEARS = [(128, 289), (64, 128), (1, 64)]


def load(path):
    """Read the file into the tensors it describes, in order."""
    raw = open(path, "rb").read()
    if raw[: len(MAGIC)] != MAGIC:
        raise ValueError(f"{path} does not start with {MAGIC!r}")
    values = np.frombuffer(raw, dtype="<f4", offset=len(MAGIC))

    at = 0

    def take(*shape):
        nonlocal at
        count = int(np.prod(shape))
        out = values[at : at + count].reshape(shape)
        at += count
        return out

    convs = [(take(o, i, 3, 3), take(o)) for o, i in CONVS]
    linears = [(take(o, i), take(o)) for o, i in LINEARS]
    if at != len(values):
        raise ValueError(f"{path} holds {len(values)} values, the layout wants {at}")
    return convs, linears


def convolve(x, weight, bias):
    """3x3 convolution, stride 1, one cell of zero padding."""
    out_channels, in_channels = weight.shape[0], weight.shape[1]
    height, width = x.shape[1], x.shape[2]
    padded = np.zeros((in_channels, height + 2, width + 2), dtype=np.float32)
    padded[:, 1:-1, 1:-1] = x

    out = np.empty((out_channels, height, width), dtype=np.float32)
    for oc in range(out_channels):
        acc = np.full((height, width), bias[oc], dtype=np.float32)
        for ic in range(in_channels):
            for ky in range(3):
                for kx in range(3):
                    acc += weight[oc, ic, ky, kx] * padded[ic, ky : ky + height, kx : kx + width]
        out[oc] = acc
    return out


def forward(weights_path, patch):
    """P(mine) for one 8x9x9 patch."""
    convs, linears = load(weights_path)

    x = np.asarray(patch, dtype=np.float32)
    for weight, bias in convs:
        x = np.maximum(convolve(x, weight, bias), 0.0)

    # Adaptive average pool to 3x3: the 9x9 map divides exactly into 3x3 blocks.
    pooled = x.reshape(x.shape[0], 3, 3, 3, 3).mean(axis=(2, 4))

    # The global mines ratio rides in channel 6, constant across the patch.
    features = np.concatenate([pooled.reshape(-1), [patch[6][4][4]]]).astype(np.float32)

    for i, (weight, bias) in enumerate(linears):
        features = weight @ features + bias
        if i < len(linears) - 1:
            features = np.maximum(features, 0.0)  # dropout is identity at eval

    return float(1.0 / (1.0 + np.exp(-features[0])))
