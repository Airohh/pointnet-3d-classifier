"""Fix the rotation-robustness limitation found in scripts/robustness.py.

The baseline model (yaw-only augmentation) collapses under arbitrary 3D
rotation (0.87 -> 0.19). This script retrains PointNet with **full SO(3)
augmentation** and re-measures rotation robustness, so we can show the fix
works. The new model is saved separately (models/pointnet_so3.pt) — the
canonical baseline is left untouched.

    python scripts/train_so3.py --epochs 15

Outputs reports/rotation_fix.json and prints a before/after table.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # import robustness.py

from pointcloud_clf import constants as C  # noqa: E402
from pointcloud_clf.data import ModelNetDataset, download_modelnet10, random_so3  # noqa: E402
from pointcloud_clf.model import PointNetClassifier  # noqa: E402
from pointcloud_clf.train import train  # noqa: E402

rb = importlib.import_module("robustness")  # accuracy()

SO3_MODEL_PATH = C.MODELS_DIR / "pointnet_so3.pt"
SO3_LABELS_PATH = C.MODELS_DIR / "labels_so3.json"


def rotation_accuracy(model_path: Path, device: str, n_trials: int = 5) -> dict:
    """Canonical vs random-SO(3) accuracy for a saved model."""
    root = download_modelnet10()
    test_ds = ModelNetDataset(root, "test", augment_train=False)
    base = np.stack([test_ds[i][0].numpy() for i in range(len(test_ds))])
    labels = np.array([test_ds[i][1] for i in range(len(test_ds))])

    model = PointNetClassifier(num_classes=len(C.CLASSES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    rng = np.random.default_rng(C.SEED)
    canonical = rb.accuracy(model, base, labels, device)
    rot = [
        rb.accuracy(model, np.stack([c @ random_so3(rng).T for c in base]), labels, device)
        for _ in range(n_trials)
    ]
    return {
        "canonical_acc": float(canonical),
        "mean_rotated_acc": float(np.mean(rot)),
        "std_rotated_acc": float(np.std(rot)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(">>> training PointNet with full SO(3) augmentation ...")
    train(
        epochs=args.epochs,
        limit_per_class=args.limit,
        use_mlflow=False,
        so3_aug=True,
        model_path=SO3_MODEL_PATH,
        labels_path=SO3_LABELS_PATH,
    )

    print("\n>>> measuring rotation robustness (baseline vs SO(3)-trained) ...")
    after = rotation_accuracy(SO3_MODEL_PATH, device)

    # Baseline numbers from the earlier robustness run, if present.
    before = {"canonical_acc": None, "mean_rotated_acc": None}
    rob_json = C.REPORTS_DIR / "robustness.json"
    if rob_json.exists():
        r = json.loads(rob_json.read_text())["rotation"]
        before = {"canonical_acc": r["canonical_acc"], "mean_rotated_acc": r["mean_rotated_acc"]}

    out = {"baseline_yaw_aug": before, "so3_aug": after}
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "rotation_fix.json").write_text(json.dumps(out, indent=2))

    print("\n=== rotation robustness: before vs after ===")
    print(f"{'':18s}{'canonical':>12s}{'random SO(3)':>14s}")
    b = before
    print(
        f"{'baseline (yaw)':18s}{_fmt(b['canonical_acc']):>12s}"
        f"{_fmt(b['mean_rotated_acc']):>14s}"
    )
    print(f"{'SO(3) aug':18s}{after['canonical_acc']:>12.3f}" f"{after['mean_rotated_acc']:>14.3f}")
    print(f"\nrotation_fix.json -> {C.REPORTS_DIR / 'rotation_fix.json'}")


def _fmt(v) -> str:
    return f"{v:.3f}" if v is not None else "n/a"


if __name__ == "__main__":
    main()
