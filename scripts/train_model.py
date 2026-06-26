"""Train PointNet on ModelNet10.

Examples:
    python scripts/train_model.py                 # full training
    python scripts/train_model.py --epochs 5 --limit 40   # fast smoke run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointcloud_clf import constants as C  # noqa: E402
from pointcloud_clf.train import train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap samples per class (fast experiments)")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          limit_per_class=args.limit, use_mlflow=not args.no_mlflow)
