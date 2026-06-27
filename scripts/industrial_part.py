"""Domain transfer demo: the pipeline is furniture-agnostic.

ModelNet10 is furniture, but mesh -> point cloud -> PointNet is the same code
for any CAD geometry. This builds a synthetic mechanical bracket (no boolean
ops, just concatenated primitives), runs it through the exact training
pipeline, and renders the sampled cloud — showing the machinery ingests
industrial parts. Classification is out-of-distribution (the model only knows
furniture), which is the honest point: swap the dataset, keep the pipeline.

    python scripts/industrial_part.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import trimesh  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointcloud_clf import constants as C  # noqa: E402
from pointcloud_clf.data import load_pointcloud, normalize  # noqa: E402
from pointcloud_clf.predict import predict_array  # noqa: E402


def build_bracket() -> trimesh.Trimesh:
    """L-bracket flange: base plate + upright wall + central hub + 4 bolt bosses."""
    parts = []
    base = trimesh.creation.box(extents=(1.0, 0.7, 0.08))
    base.apply_translation((0, 0, 0.04))
    parts.append(base)
    wall = trimesh.creation.box(extents=(0.08, 0.7, 0.6))
    wall.apply_translation((-0.46, 0, 0.34))
    parts.append(wall)
    hub = trimesh.creation.cylinder(radius=0.16, height=0.18, sections=48)
    hub.apply_translation((0.12, 0, 0.13))
    parts.append(hub)
    for sx in (-0.32, 0.32):
        for sy in (-0.24, 0.24):
            boss = trimesh.creation.cylinder(radius=0.05, height=0.14, sections=24)
            boss.apply_translation((sx, sy, 0.11))
            parts.append(boss)
    return trimesh.util.concatenate(parts)


def main() -> None:
    if not C.MODEL_PATH.exists():
        raise SystemExit("no trained model; run scripts/train_model.py first")

    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stl = C.REPORTS_DIR / "_industrial_part.stl"
    build_bracket().export(stl)

    pts = normalize(load_pointcloud(stl, C.NUM_POINTS))
    res = predict_array(pts)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, c=pts[:, 2], cmap="cividis")
    ax.set_title(
        "industrial CAD part (synthetic bracket)\nsame pipeline -> 1024-pt cloud",
        fontsize=11,
    )
    ax.set_axis_off()
    ax.view_init(elev=24, azim=-125)
    ax.set_box_aspect((1, 0.7, 0.6))
    fig.tight_layout()
    out = C.REPORTS_DIR / "industrial_part.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    top3 = [(r["label"], round(r["probability"], 3)) for r in res["top_k"]]
    print(f"industrial part -> {out}")
    print(f"OOD prediction (furniture-trained model): {top3}")


if __name__ == "__main__":
    main()
