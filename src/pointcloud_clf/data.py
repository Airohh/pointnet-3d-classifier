"""Data pipeline: download ModelNet10, sample point clouds from meshes,
normalise, augment, and serve as a PyTorch Dataset.

The expensive step (mesh -> fixed-size point cloud) is cached to .npy so that
training epochs do not re-sample the surface every time.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

from . import constants as C


# --- Mesh -> point cloud -----------------------------------------------------
def sample_mesh(
    mesh: trimesh.Trimesh, n_points: int = C.NUM_POINTS, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Uniformly sample `n_points` from a mesh surface (area-weighted)."""
    rng = rng or np.random.default_rng()
    pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=int(rng.integers(1 << 31)))
    return np.asarray(pts, dtype=np.float32)


def load_pointcloud(
    path: str | Path, n_points: int = C.NUM_POINTS, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Load any mesh file (.off/.ply/.stl/.obj) -> (n_points, 3) array."""
    mesh = trimesh.load(path, force="mesh")
    if mesh.vertices.shape[0] == 0:
        raise ValueError(f"empty mesh: {path}")
    return sample_mesh(mesh, n_points, rng)


def normalize(points: np.ndarray) -> np.ndarray:
    """Center on centroid and scale into the unit sphere (scale invariance)."""
    points = points - points.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale
    return points.astype(np.float32)


def random_so3(rng: np.random.Generator) -> np.ndarray:
    """Uniform random 3D rotation matrix (QR of a Gaussian, sign-fixed)."""
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


def augment(points: np.ndarray, rng: np.random.Generator, so3: bool = False) -> np.ndarray:
    """Train-time augmentation: rotation + small jitter.

    so3=False -> yaw only (rotation about the up axis); the default, matches
    how CAD parts usually sit. so3=True -> arbitrary 3D rotation, which makes
    the model robust to any scan orientation (see scripts/train_so3.py).
    """
    if so3:
        rot = random_so3(rng)
    else:
        theta = rng.uniform(0, 2 * np.pi)
        cos, sin = np.cos(theta), np.sin(theta)
        rot = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)
    points = points @ rot.T
    points = points + rng.normal(0, 0.02, size=points.shape).astype(np.float32)
    return points.astype(np.float32)


# --- Download ----------------------------------------------------------------
def download_modelnet10(dest: Path = C.RAW_DIR) -> Path:
    """Download + extract ModelNet10 if not already present."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest / "ModelNet10"
    if root.exists():
        return root
    print(f"Downloading ModelNet10 from {C.MODELNET10_URL} ...")
    with urllib.request.urlopen(C.MODELNET10_URL, timeout=120) as resp:
        blob = resp.read()
    print(f"Extracting {len(blob) / 1e6:.0f} MB ...")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(dest)
    return root


def list_split(root: Path, split: str) -> list[tuple[Path, int]]:
    """Return (mesh_path, label_index) pairs for the given split (train/test)."""
    items: list[tuple[Path, int]] = []
    for label, cls in enumerate(C.CLASSES):
        for off in sorted((root / cls / split).glob("*.off")):
            items.append((off, label))
    return items


# --- Dataset -----------------------------------------------------------------
class ModelNetDataset(Dataset):
    """Point-cloud dataset with on-disk caching of the sampled clouds."""

    def __init__(
        self,
        root: Path,
        split: str,
        n_points: int = C.NUM_POINTS,
        augment_train: bool = True,
        limit_per_class: int | None = None,
        so3_aug: bool = False,
    ):
        self.split = split
        self.n_points = n_points
        self.augment_train = augment_train and split == "train"
        self.so3_aug = so3_aug
        self.rng = np.random.default_rng(C.SEED)
        items = list_split(root, split)
        if limit_per_class is not None:
            counts: dict[int, int] = {}
            kept = []
            for path, label in items:
                if counts.get(label, 0) < limit_per_class:
                    kept.append((path, label))
                    counts[label] = counts.get(label, 0) + 1
            items = kept
        self.items = items
        self.cache_dir = C.CACHE_DIR / split
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.items)

    def _cached_cloud(self, idx: int, path: Path) -> np.ndarray:
        cache = self.cache_dir / f"{path.stem}_{self.n_points}.npy"
        if cache.exists():
            return np.load(cache)
        cloud = normalize(load_pointcloud(path, self.n_points, self.rng))
        np.save(cache, cloud)
        return cloud

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        cloud = self._cached_cloud(idx, path)
        if self.augment_train:
            cloud = augment(cloud, self.rng, so3=self.so3_aug)
        return torch.from_numpy(cloud).float(), label
