"""Training loop with MLflow tracking and an honest train/test evaluation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import constants as C
from .data import ModelNetDataset, download_modelnet10
from .model import PointNetClassifier, feature_transform_regularizer


def set_seed(seed: int = C.SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: PointNetClassifier, loader: DataLoader, device: str):
    model.eval()
    correct = total = 0
    per_class_correct: dict[int, int] = {}
    per_class_total: dict[int, int] = {}
    for clouds, labels in loader:
        clouds, labels = clouds.to(device), labels.to(device)
        logits, _ = model(clouds)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        for y, p in zip(labels.tolist(), pred.tolist()):
            per_class_total[y] = per_class_total.get(y, 0) + 1
            per_class_correct[y] = per_class_correct.get(y, 0) + int(y == p)
    acc = correct / max(total, 1)
    per_class = {
        C.CLASSES[k]: per_class_correct.get(k, 0) / max(per_class_total.get(k, 1), 1)
        for k in range(len(C.CLASSES))
    }
    return acc, per_class


def train(epochs: int = C.EPOCHS, batch_size: int = C.BATCH_SIZE,
          lr: float = C.LR, limit_per_class: int | None = None,
          use_mlflow: bool = True, so3_aug: bool = False,
          model_path: Path | None = None,
          labels_path: Path | None = None,
          feature_transform: bool = True,
          augment_train: bool = True) -> dict:
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = download_modelnet10()
    model_path = model_path or C.MODEL_PATH
    labels_path = labels_path or C.LABELS_PATH

    train_ds = ModelNetDataset(root, "train", limit_per_class=limit_per_class,
                               so3_aug=so3_aug, augment_train=augment_train)
    test_ds = ModelNetDataset(root, "test", augment_train=False,
                              limit_per_class=limit_per_class)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)

    model = PointNetClassifier(num_classes=len(C.CLASSES),
                               feature_transform=feature_transform).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    mlf = _maybe_mlflow(use_mlflow)
    if mlf:
        mlf.start_run()
        mlf.log_params({"epochs": epochs, "batch_size": batch_size, "lr": lr,
                        "num_points": C.NUM_POINTS, "train_n": len(train_ds),
                        "test_n": len(test_ds)})

    best_acc = 0.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for clouds, labels in train_loader:
            clouds, labels = clouds.to(device), labels.to(device)
            opt.zero_grad()
            logits, t_feat = model(clouds)
            loss = F.cross_entropy(logits, labels)
            loss = loss + C.FEATURE_TRANSFORM_REG * feature_transform_regularizer(t_feat)
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()

        acc, per_class = evaluate(model, test_loader, device)
        dt = time.time() - t0
        avg_loss = running / max(len(train_loader), 1)
        print(f"epoch {epoch:02d}  loss {avg_loss:.3f}  test_acc {acc:.3f}  ({dt:.0f}s)")
        history.append({"epoch": epoch, "loss": avg_loss, "test_acc": acc})
        if mlf:
            mlf.log_metrics({"train_loss": avg_loss, "test_acc": acc}, step=epoch)

        if acc > best_acc:
            best_acc = acc
            _save(model, per_class, model_path, labels_path)

    if mlf:
        mlf.log_metric("best_test_acc", best_acc)
        mlf.end_run()

    Path(C.REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best test accuracy: {best_acc:.3f}  -> {model_path}")
    return {"best_acc": best_acc, "history": history}


def _save(model: PointNetClassifier, per_class: dict,
          model_path: Path = C.MODEL_PATH,
          labels_path: Path = C.LABELS_PATH) -> None:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    labels_path.write_text(json.dumps(
        {"classes": list(C.CLASSES), "num_points": C.NUM_POINTS,
         "per_class_acc": per_class}, indent=2))


def _maybe_mlflow(enabled: bool):
    if not enabled:
        return None
    try:
        import mlflow
        mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
        return mlflow
    except Exception as exc:  # pragma: no cover
        print(f"mlflow disabled: {exc}")
        return None
