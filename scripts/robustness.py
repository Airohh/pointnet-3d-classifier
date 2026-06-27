"""Robustness benchmark: how does the trained PointNet hold up under the
conditions a *real* 3D scan presents — sensor noise, partial/occluded surfaces,
and arbitrary orientation?

A clean test-set number (one figure on holdout CAD meshes) says nothing about
behaviour in production, where parts are scanned noisily, seen from one side,
and arrive in any pose. This script perturbs each test cloud and re-measures
accuracy, sweeping three axes:

    1. Gaussian noise   — additive sensor jitter, sigma in unit-sphere units.
    2. Occlusion        — drop a spatial cap of points (one-sided scan), then
                          resample to keep the input size fixed.
    3. SO(3) rotation   — arbitrary 3D rotation. The model was trained with
                          yaw-only augmentation, so full rotation is the honest
                          out-of-distribution stress test (a known PointNet
                          limitation: the input T-Net helps but does not make
                          the network rotation-invariant).

Outputs reports/robustness.json and reports/robustness.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointcloud_clf import constants as C  # noqa: E402
from pointcloud_clf.data import ModelNetDataset, download_modelnet10, random_so3  # noqa: E402
from pointcloud_clf.model import PointNetClassifier  # noqa: E402

NOISE_SIGMAS = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
OCCLUSION_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_ROTATION_TRIALS = 5


# --- Perturbations (operate on a single (N, 3) normalised cloud) -------------
def add_noise(pts: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma == 0.0:
        return pts
    return (pts + rng.normal(0, sigma, size=pts.shape)).astype(np.float32)


def occlude(pts: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Remove a spatial cap (the `frac` points farthest along a random
    direction — a one-sided scan), then resample with replacement so the
    input keeps NUM_POINTS. frac=0 is a no-op."""
    if frac == 0.0:
        return pts
    n = pts.shape[0]
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    keep = n - int(round(frac * n))
    order = np.argsort(pts @ direction)  # ascending projection
    kept = pts[order[:keep]]  # drop the far cap
    fill = rng.integers(0, keep, size=n - keep)  # resample to full size
    return np.concatenate([kept, kept[fill]], axis=0).astype(np.float32)


# --- Eval over a perturbed copy of the whole test set ------------------------
def accuracy(model, clouds: np.ndarray, labels: np.ndarray, device: str) -> float:
    ds = TensorDataset(torch.from_numpy(clouds).float(), torch.from_numpy(labels).long())
    loader = DataLoader(ds, batch_size=C.BATCH_SIZE)
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))[0].argmax(dim=1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def main() -> None:
    if not C.MODEL_PATH.exists():
        raise SystemExit("no trained model; run scripts/train_model.py first")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(C.SEED)

    # Load the cached, normalised test clouds once into memory.
    root = download_modelnet10()
    test_ds = ModelNetDataset(root, "test", augment_train=False)
    base = np.stack([test_ds[i][0].numpy() for i in range(len(test_ds))])  # (M,N,3)
    labels = np.array([test_ds[i][1] for i in range(len(test_ds))])
    print(f"test clouds: {base.shape[0]}")

    model = PointNetClassifier(num_classes=len(C.CLASSES)).to(device)
    model.load_state_dict(torch.load(C.MODEL_PATH, map_location=device))

    results: dict = {"noise": {}, "occlusion": {}, "rotation": {}}

    print("\n[noise sweep]")
    for sigma in NOISE_SIGMAS:
        pert = np.stack([add_noise(c, sigma, rng) for c in base])
        acc = accuracy(model, pert, labels, device)
        results["noise"][f"{sigma:.2f}"] = acc
        print(f"  sigma {sigma:.2f}  acc {acc:.3f}")

    print("\n[occlusion sweep]")
    for frac in OCCLUSION_FRACS:
        pert = np.stack([occlude(c, frac, rng) for c in base])
        acc = accuracy(model, pert, labels, device)
        results["occlusion"][f"{frac:.1f}"] = acc
        print(f"  drop {frac:.0%}  acc {acc:.3f}")

    print("\n[SO(3) rotation]")
    clean = results["noise"]["0.00"]
    rot_accs = []
    for t in range(N_ROTATION_TRIALS):
        pert = np.stack([c @ random_so3(rng).T for c in base])
        rot_accs.append(accuracy(model, pert, labels, device))
    results["rotation"] = {
        "canonical_acc": clean,
        "mean_rotated_acc": float(np.mean(rot_accs)),
        "std_rotated_acc": float(np.std(rot_accs)),
        "trials": N_ROTATION_TRIALS,
    }
    print(
        f"  canonical {clean:.3f}  ->  random SO(3) "
        f"{np.mean(rot_accs):.3f} +/- {np.std(rot_accs):.3f}"
    )

    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "robustness.json").write_text(json.dumps(results, indent=2))
    _plot(results)


def _plot(results: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped robustness plot")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    xs = [float(k) for k in results["noise"]]
    ax1.plot(xs, list(results["noise"].values()), "o-", color="#c0392b")
    ax1.set_xlabel("Gaussian noise sigma (unit-sphere)")
    ax1.set_ylabel("test accuracy")
    ax1.set_title("Sensor noise")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    xs = [float(k) for k in results["occlusion"]]
    ax2.plot(xs, list(results["occlusion"].values()), "s-", color="#2471a3")
    ax2.set_xlabel("fraction of surface occluded")
    ax2.set_title("Partial / one-sided scan")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)

    r = results["rotation"]
    fig.suptitle(
        f"PointNet robustness  |  SO(3) rotation: "
        f"{r['canonical_acc']:.2f} (canonical) -> {r['mean_rotated_acc']:.2f} (rotated)",
        fontsize=11,
    )
    fig.tight_layout()
    out = C.REPORTS_DIR / "robustness.png"
    fig.savefig(out, dpi=120)
    print(f"robustness plot -> {out}")


if __name__ == "__main__":
    main()
