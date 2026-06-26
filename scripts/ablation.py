"""Ablation study: does each PointNet component earn its place?

Retrains the network under three configurations at identical epochs/seed and
reports best test accuracy, so the contribution of each piece is measurable
rather than assumed:

    full            feature T-Net + train-time augmentation   (the baseline)
    no_feat_tnet    drop the 64x64 feature transform
    no_aug          drop yaw + jitter augmentation

Temp checkpoints go to models/_ablation_*.pt; the canonical models/pointnet.pt
is left untouched. Writes reports/ablation.json.

    python scripts/ablation.py --epochs 15
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointcloud_clf import constants as C  # noqa: E402
from pointcloud_clf.train import train  # noqa: E402

CONFIGS = {
    "full": {"feature_transform": True, "augment_train": True},
    "no_feat_tnet": {"feature_transform": False, "augment_train": True},
    "no_aug": {"feature_transform": True, "augment_train": False},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    results = {}
    for name, cfg in CONFIGS.items():
        print(f"\n===== ablation: {name}  {cfg} =====")
        out = train(
            epochs=args.epochs,
            limit_per_class=args.limit,
            use_mlflow=False,
            model_path=C.MODELS_DIR / f"_ablation_{name}.pt",
            labels_path=C.MODELS_DIR / f"_ablation_{name}.json",
            **cfg,
        )
        results[name] = {"best_acc": out["best_acc"], **cfg}
        print(f"  -> {name}: best_acc {out['best_acc']:.3f}")

    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "ablation.json").write_text(json.dumps(results, indent=2))

    print("\n=== ablation summary ===")
    print(f"{'config':16s}{'best_acc':>10s}")
    for name, r in results.items():
        print(f"{name:16s}{r['best_acc']:>10.3f}")
    print(f"\nablation.json -> {C.REPORTS_DIR / 'ablation.json'}")


if __name__ == "__main__":
    main()
