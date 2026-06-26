"""Inference performance benchmark: latency, throughput, model size.

Industrialisation needs numbers an application owner cares about — how fast is
a prediction, how does it scale with batch size, how big is the model. Runs on
CPU (the deployment target here). Writes reports/benchmark.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointcloud_clf import constants as C  # noqa: E402
from pointcloud_clf.model import PointNetClassifier  # noqa: E402

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
WARMUP = 3
ITERS = 20


def main() -> None:
    torch.manual_seed(C.SEED)
    device = "cpu"
    model = PointNetClassifier(num_classes=len(C.CLASSES)).to(device)
    if C.MODEL_PATH.exists():
        model.load_state_dict(torch.load(C.MODEL_PATH, map_location=device))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    size_mb = (C.MODEL_PATH.stat().st_size / 1e6) if C.MODEL_PATH.exists() else None

    print(
        f"params: {n_params:,}   model file: " f"{size_mb:.1f} MB"
        if size_mb
        else f"params: {n_params:,}"
    )
    print(f"\n{'batch':>6} {'ms/batch':>10} {'ms/sample':>11} {'samples/s':>11}")

    results = {
        "n_params": n_params,
        "model_size_mb": size_mb,
        "threads": torch.get_num_threads(),
        "by_batch": {},
    }
    for bs in BATCH_SIZES:
        x = torch.randn(bs, C.NUM_POINTS, 3, device=device)
        with torch.no_grad():
            for _ in range(WARMUP):
                model(x)
            t0 = time.perf_counter()
            for _ in range(ITERS):
                model(x)
            dt = (time.perf_counter() - t0) / ITERS
        ms_batch = dt * 1e3
        ms_sample = ms_batch / bs
        sps = bs / dt
        results["by_batch"][str(bs)] = {
            "ms_per_batch": round(ms_batch, 2),
            "ms_per_sample": round(ms_sample, 2),
            "samples_per_s": round(sps, 1),
        }
        print(f"{bs:>6} {ms_batch:>10.2f} {ms_sample:>11.2f} {sps:>11.1f}")

    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "benchmark.json").write_text(json.dumps(results, indent=2))
    print(f"\nbenchmark.json -> {C.REPORTS_DIR / 'benchmark.json'}")


if __name__ == "__main__":
    main()
