"""Export the trained PatchCNN to ONNX and validate with onnxruntime.

Usage:
    python neural/export.py [--checkpoint neural/checkpoints/best.pt]
                            [--output neural/onnx/model.onnx]
                            [--opset 17]

Asserts that max absolute difference between PyTorch and ONNX outputs is < 1e-4,
and that the result is one self-contained file.

That second check exists because torch writes the weights to a sidecar
`model.onnx.data` by default, leaving a 6 KB graph next to a 485 KB blob. Both
onnxruntime and tract find the sidecar as long as it is in the same directory, so
everything appears to work — right up until the `.onnx` is committed, moved or
copied on its own, at which point it is a model with no weights in it.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
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
    # Fold any sidecar weights back into the file itself.
    model_proto = onnx.load(str(output))  # merges external data, if there is any
    onnx.save(model_proto, str(output), save_as_external_data=False)
    sidecar = output.with_name(output.name + ".data")
    if sidecar.exists():
        sidecar.unlink()

    stray = [
        initializer.name
        for initializer in model_proto.graph.initializer
        if initializer.data_location != onnx.TensorProto.DEFAULT
    ]
    assert not stray, f"weights still held outside the file: {stray}"

    weights = sum(len(i.raw_data) for i in model_proto.graph.initializer)
    size = output.stat().st_size
    print(f"Exported ONNX model to {output} ({size:,} bytes)")
    assert size > weights, (
        f"{output} is {size:,} bytes but holds {weights:,} bytes of weights — "
        "the file does not contain its own parameters"
    )

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
