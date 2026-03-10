"""Export the trained PatchCNN to ONNX and validate with onnxruntime.

Usage:
    python neural/export.py [--checkpoint neural/checkpoints/best.pt]
                            [--output neural/onnx/model.onnx]
                            [--opset 17]

Asserts that max absolute difference between PyTorch and ONNX outputs is < 1e-4.
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from model import PatchCNN

CKPT_DEFAULT = Path(__file__).parent / "checkpoints" / "best.pt"
ONNX_DEFAULT = Path(__file__).parent / "onnx" / "model.onnx"


def export(checkpoint: Path, output: Path, opset: int):
    output.parent.mkdir(parents=True, exist_ok=True)

    model = PatchCNN()
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Dynamic batch axis so the Rust side can send any number of hidden cells.
    dummy = torch.zeros(1, 8, 9, 9)

    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["patch"],
        output_names=["prob"],
        dynamic_axes={"patch": {0: "batch"}, "prob": {0: "batch"}},
        opset_version=opset,
    )
    print(f"Exported ONNX model to {output}")

    # --- Validate ---
    with torch.no_grad():
        torch_out = model(dummy).numpy()

    sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["prob"], {"patch": dummy.numpy()})[0]

    max_diff = np.abs(torch_out - ort_out).max()
    print(f"Max abs diff (torch vs onnxruntime): {max_diff:.2e}")
    assert max_diff < 1e-4, f"ONNX round-trip error too large: {max_diff}"
    print("Validation passed.")

    # Larger random batch
    big = torch.randn(64, 8, 9, 9)
    with torch.no_grad():
        t_big = model(big).numpy()
    o_big = sess.run(["prob"], {"patch": big.numpy()})[0]
    max_diff2 = np.abs(t_big - o_big).max()
    print(f"Max abs diff (batch=64): {max_diff2:.2e}")
    assert max_diff2 < 1e-4, f"Batch validation error: {max_diff2}"
    print("Batch validation passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CKPT_DEFAULT)
    parser.add_argument("--output", type=Path, default=ONNX_DEFAULT)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found at {args.checkpoint}")
        raise SystemExit(1)

    export(args.checkpoint, args.output, args.opset)


if __name__ == "__main__":
    main()
