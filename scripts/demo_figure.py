"""End-to-end demo figure: take a real ModelNet10 mesh, sample it into the
point cloud the model actually sees, run PointNet, and render the cloud titled
with the prediction. Reliable (matplotlib, no WebGL) — used for the README.

    python scripts/demo_figure.py                 # default: a chair
    python scripts/demo_figure.py --mesh path.off # any .off/.ply/.stl/.obj
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointcloud_clf import constants as C  # noqa: E402
from pointcloud_clf.data import load_pointcloud, normalize  # noqa: E402
from pointcloud_clf.predict import predict_array  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mesh",
        type=Path,
        default=C.RAW_DIR / "ModelNet10" / "chair" / "test" / "chair_0890.off",
    )
    ap.add_argument("--out", type=Path, default=C.REPORTS_DIR / "demo_pointcloud.png")
    args = ap.parse_args()

    if not C.MODEL_PATH.exists():
        raise SystemExit("no trained model; run scripts/train_model.py first")

    pts = normalize(load_pointcloud(args.mesh, C.NUM_POINTS))
    res = predict_array(pts)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=3, c=pts[:, 1], cmap="viridis")
    ax.set_title(
        f"input: {args.mesh.name}\n"
        f"PointNet -> {res['prediction']} ({res['confidence'] * 100:.1f}%)",
        fontsize=11,
    )
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-60)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    top3 = [(r["label"], round(r["probability"], 3)) for r in res["top_k"]]
    print(f"demo figure -> {args.out}  | top-3: {top3}")


if __name__ == "__main__":
    main()
