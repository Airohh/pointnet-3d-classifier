"""Out-of-distribution / calibration probe.

A softmax classifier is forced to pick one of its 10 classes for *any* input —
even a shape that belongs to none of them. This script feeds geometric
primitives that are not ModelNet10 furniture (sphere, torus, cone, cylinder)
and prints the model's top prediction + confidence. The takeaway: the network
is often **confidently wrong** on OOD input, so a deployment needs an
abstain/threshold mechanism (or an explicit "unknown" class), not raw softmax.

    python scripts/ood_demo.py

Writes reports/ood.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointcloud_clf import constants as C  # noqa: E402
from pointcloud_clf.data import sample_mesh  # noqa: E402
from pointcloud_clf.predict import predict_array  # noqa: E402

OOD_SHAPES = {
    "sphere": trimesh.creation.icosphere(subdivisions=3, radius=1.0),
    "torus": trimesh.creation.torus(major_radius=1.0, minor_radius=0.3),
    "cone": trimesh.creation.cone(radius=1.0, height=2.0),
    "cylinder": trimesh.creation.cylinder(radius=0.6, height=2.0),
}


def main() -> None:
    if not C.MODEL_PATH.exists():
        raise SystemExit("no trained model; run scripts/train_model.py first")

    rng = np.random.default_rng(C.SEED)
    results = {}
    print(f"{'OOD shape':12s}{'predicted':14s}{'confidence':>11s}")
    for name, mesh in OOD_SHAPES.items():
        pts = sample_mesh(mesh, C.NUM_POINTS, rng)
        out = predict_array(pts)
        results[name] = {"prediction": out["prediction"], "confidence": out["confidence"]}
        print(f"{name:12s}{out['prediction']:14s}{out['confidence']:>11.3f}")

    confs = [r["confidence"] for r in results.values()]
    print(f"\nmean confidence on OOD shapes: {np.mean(confs):.3f}")
    print(
        "=> softmax is overconfident on inputs outside the 10 training classes;"
        "\n   a production system should threshold/abstain, not trust raw confidence."
    )

    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "ood.json").write_text(json.dumps(results, indent=2))
    print(f"\nood.json -> {C.REPORTS_DIR / 'ood.json'}")


if __name__ == "__main__":
    main()
