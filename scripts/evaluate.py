"""Evaluate the trained model on the test split: overall + per-class accuracy
and a confusion matrix saved to reports/confusion_matrix.png."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointcloud_clf import constants as C  # noqa: E402
from pointcloud_clf.data import ModelNetDataset, download_modelnet10  # noqa: E402
from pointcloud_clf.model import PointNetClassifier  # noqa: E402
from pointcloud_clf.train import evaluate  # noqa: E402


def main() -> None:
    if not C.MODEL_PATH.exists():
        raise SystemExit("no trained model; run scripts/train_model.py first")
    root = download_modelnet10()
    test_ds = ModelNetDataset(root, "test", augment_train=False)
    loader = DataLoader(test_ds, batch_size=C.BATCH_SIZE)

    model = PointNetClassifier(num_classes=len(C.CLASSES))
    model.load_state_dict(torch.load(C.MODEL_PATH, map_location="cpu"))

    acc, per_class = evaluate(model, loader, "cpu")
    print(f"overall test accuracy: {acc:.3f}")
    for cls, a in sorted(per_class.items(), key=lambda kv: kv[1]):
        print(f"  {cls:12s} {a:.3f}")

    # Confusion matrix
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for clouds, labels in loader:
            pred = model(clouds)[0].argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(pred.tolist())
    cm = np.zeros((len(C.CLASSES), len(C.CLASSES)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "eval.json").write_text(
        json.dumps({"overall_acc": acc, "per_class_acc": per_class}, indent=2)
    )
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(C.CLASSES)), C.CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(len(C.CLASSES)), C.CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for i in range(len(C.CLASSES)):
            for j in range(len(C.CLASSES)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(C.REPORTS_DIR / "confusion_matrix.png", dpi=120)
        print(f"confusion matrix -> {C.REPORTS_DIR / 'confusion_matrix.png'}")
    except ImportError:
        print("matplotlib not installed; skipped confusion-matrix plot")


if __name__ == "__main__":
    main()
