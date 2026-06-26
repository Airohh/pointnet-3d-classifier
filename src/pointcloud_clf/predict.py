"""Inference helper: load the trained model once, classify a mesh file."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import constants as C
from .data import load_pointcloud, normalize
from .model import PointNetClassifier


@lru_cache(maxsize=1)
def _load_model() -> tuple[PointNetClassifier, list[str], int]:
    meta = json.loads(C.LABELS_PATH.read_text())
    classes = meta["classes"]
    n_points = meta.get("num_points", C.NUM_POINTS)
    model = PointNetClassifier(num_classes=len(classes))
    model.load_state_dict(torch.load(C.MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, classes, n_points


def predict_array(points: np.ndarray, top_k: int = 3) -> dict:
    """Classify a raw (N, 3) point cloud array."""
    model, classes, _ = _load_model()
    cloud = normalize(np.asarray(points, dtype=np.float32))
    x = torch.from_numpy(cloud).float().unsqueeze(0)   # (1, N, 3)
    with torch.no_grad():
        logits, _ = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
    k = min(top_k, len(classes))
    top = torch.topk(probs, k)
    ranked = [{"label": classes[i], "probability": round(probs[i].item(), 4)}
              for i in top.indices.tolist()]
    return {"prediction": ranked[0]["label"],
            "confidence": ranked[0]["probability"],
            "top_k": ranked}


def predict_file(path: str | Path, top_k: int = 3) -> dict:
    """Classify any mesh file (.off/.ply/.stl/.obj)."""
    _, _, n_points = _load_model()
    points = load_pointcloud(path, n_points)
    return predict_array(points, top_k=top_k)
