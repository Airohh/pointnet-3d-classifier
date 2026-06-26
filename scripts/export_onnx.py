"""Export the trained PointNet to ONNX and verify torch/ONNX parity.

ONNX decouples the model from PyTorch: any application (C++, C#, a Node
backend, an edge runtime) can run inference via ONNX Runtime without a Python
or torch dependency — the practical meaning of "interface a model with a
software application". This script exports models/pointnet.pt to
models/pointnet.onnx, runs the same input through both, and asserts the outputs
match within tolerance.

    python scripts/export_onnx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointcloud_clf import constants as C  # noqa: E402
from pointcloud_clf.model import PointNetClassifier  # noqa: E402

ONNX_PATH = C.MODELS_DIR / "pointnet.onnx"


class _LogitsOnly(torch.nn.Module):
    """Wrap the classifier so ONNX export has a single tensor output
    (the model normally also returns the feature-transform matrix)."""

    def __init__(self, model: PointNetClassifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[0]


def main() -> None:
    if not C.MODEL_PATH.exists():
        raise SystemExit("no trained model; run scripts/train_model.py first")

    model = PointNetClassifier(num_classes=len(C.CLASSES))
    model.load_state_dict(torch.load(C.MODEL_PATH, map_location="cpu"))
    model.eval()
    wrapped = _LogitsOnly(model).eval()

    dummy = torch.randn(1, C.NUM_POINTS, 3)
    torch.onnx.export(
        wrapped,
        dummy,
        str(ONNX_PATH),
        input_names=["points"],
        output_names=["logits"],
        dynamic_axes={"points": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"exported -> {ONNX_PATH}  ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")

    # --- Parity check: torch vs ONNX Runtime ---
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(ONNX_PATH)))

    rng = np.random.default_rng(C.SEED)
    x = rng.standard_normal((4, C.NUM_POINTS, 3)).astype(np.float32)
    with torch.no_grad():
        torch_out = wrapped(torch.from_numpy(x)).numpy()

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["logits"], {"points": x})[0]

    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    agree = bool((torch_out.argmax(1) == onnx_out.argmax(1)).all())
    print(f"max |torch - onnx| = {max_diff:.2e}   argmax agree: {agree}")
    if max_diff > 1e-3 or not agree:
        raise SystemExit(f"PARITY FAIL: max_diff={max_diff:.2e} agree={agree}")
    print("parity OK — ONNX output matches PyTorch within 1e-3")

    _latency(wrapped, sess)


def _latency(wrapped, sess, iters: int = 30, warmup: int = 5) -> None:
    """Single-sample CPU latency: PyTorch vs ONNX Runtime."""
    import time

    x = np.random.default_rng(0).standard_normal((1, C.NUM_POINTS, 3)).astype(np.float32)
    xt = torch.from_numpy(x)

    with torch.no_grad():
        for _ in range(warmup):
            wrapped(xt)
        t0 = time.perf_counter()
        for _ in range(iters):
            wrapped(xt)
        torch_ms = (time.perf_counter() - t0) / iters * 1e3

    for _ in range(warmup):
        sess.run(["logits"], {"points": x})
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(["logits"], {"points": x})
    onnx_ms = (time.perf_counter() - t0) / iters * 1e3

    speedup = torch_ms / onnx_ms if onnx_ms else float("nan")
    print(
        f"\nlatency (batch=1, CPU): torch {torch_ms:.2f} ms  |  "
        f"onnxruntime {onnx_ms:.2f} ms  ({speedup:.2f}x)"
    )


if __name__ == "__main__":
    main()
