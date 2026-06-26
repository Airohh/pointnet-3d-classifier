"""Project-wide constants and paths."""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODEL_PATH = MODELS_DIR / "pointnet.pt"
LABELS_PATH = MODELS_DIR / "labels.json"

# --- Dataset -----------------------------------------------------------------
# ModelNet10: 10 categories of CAD meshes (.off). The pipeline is domain
# agnostic: any watertight mesh of an industrial part can be classified the
# same way once the network is trained on the relevant catalogue.
MODELNET10_URL = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"

CLASSES = (
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
)

# --- Point sampling ----------------------------------------------------------
NUM_POINTS = 1024          # points sampled per mesh surface
NORMALIZE = True           # center + unit-sphere scale (scale invariance)

# --- Training ----------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
FEATURE_TRANSFORM_REG = 1e-3   # weight of the T-Net orthogonality penalty

MLFLOW_EXPERIMENT = "pointcloud-classifier"
