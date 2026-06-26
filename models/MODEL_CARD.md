# Model Card — PointNet / ModelNet10

## Overview
- **Task**: 10-class 3D shape classification from geometry alone (no metadata).
- **Architecture**: PointNet (Qi et al., CVPR 2017) — input 3×3 T-Net, shared
  per-point MLPs, 64×64 feature T-Net, global max-pool, FC head. 3.46 M params.
- **Input**: 1024 points sampled (area-weighted) from a mesh surface, centered +
  scaled to the unit sphere.
- **Output**: softmax over 10 classes + top-k.
- **Artifacts**: `models/pointnet.pt` (PyTorch), `models/pointnet.onnx` (ONNX,
  parity-checked).

## Training data
- **ModelNet10** — CAD meshes, 10 furniture categories.
  ~3991 train / 908 test meshes. Source:
  `http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip`.
- Augmentation: random yaw rotation + Gaussian jitter (σ=0.02).

## Metrics (test split, 908 meshes)
- **Overall accuracy: 0.872.**
- Per-class: chair 0.99, sofa 0.97, bed 0.96, monitor 0.94, toilet 0.93,
  table 0.86, dresser 0.85, bathtub 0.80, **desk 0.66, night_stand 0.66**.
- Confusion concentrates on geometric near-duplicates (desk↔table,
  night_stand↔dresser). See `reports/confusion_matrix.png`.

## Robustness (see reports/robustness.*)
- **Noise** (sensor jitter): graceful — 0.872 → 0.66 at σ=0.10.
- **Occlusion** (partial scan): 0.872 → 0.38 at 50 % points dropped.
- **Rotation**: arbitrary SO(3) collapses the yaw-trained model to **0.19**.
  Retraining with full SO(3) augmentation recovers rotated accuracy to **0.59**
  (canonical drops to 0.64) — `models/pointnet_so3.pt`, `reports/rotation_fix.json`.

## Known limitations
- **Not rotation-invariant** unless trained with SO(3) augmentation; the input
  T-Net mitigates but does not solve this.
- Weak on box-like classes that share silhouettes (desk/table, night_stand/dresser).
- Trained on clean CAD meshes; real noisy/partial scans degrade accuracy (quantified above).
- Furniture domain only; retraining required for other catalogues (pipeline is
  category-agnostic).

## Performance
- 3.46 M params, 13.9 MB. ~4 ms/sample on CPU (`reports/benchmark.json`).

## Reproducibility
- Seed 42 (`constants.SEED`). 15 epochs, Adam lr 1e-3, weight decay 1e-4,
  StepLR (γ=0.5 every 10), feature-transform reg 1e-3, batch 32.
- Versions: PyTorch 2.9 (CPU), trimesh 4.12, onnxruntime 1.27.
- Reproduce: `python scripts/train_model.py --epochs 15` then
  `python scripts/evaluate.py`.

## Intended use
Portfolio / demonstration of the path *raw 3D geometry → robust model →
production API*. Not validated for safety-critical industrial deployment.
