"""PointNet for point-cloud classification.

Reference: Qi et al., "PointNet: Deep Learning on Point Sets for 3D
Classification and Segmentation" (CVPR 2017). This is a faithful, compact
re-implementation: two spatial transformer networks (input 3x3 + feature
64x64), shared per-point MLPs, a symmetric max-pool aggregator that makes the
network invariant to point ordering, and a classification head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Predicts a k x k affine matrix that aligns the input to a canonical pose."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(64), nn.BatchNorm1d(128), nn.BatchNorm1d(1024)
        self.bn4, self.bn5 = nn.BatchNorm1d(512), nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, k, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2)[0]  # symmetric pool -> (B, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (B, k*k)
        identity = torch.eye(self.k, device=x.device).flatten()
        return (x + identity).view(-1, self.k, self.k)


class PointNetEncoder(nn.Module):
    """Per-point feature extraction + global max-pool. Returns global feature
    and the feature-transform matrix (for the orthogonality regulariser)."""

    def __init__(self, feature_transform: bool = True):
        super().__init__()
        self.feature_transform = feature_transform
        self.input_tnet = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(64), nn.BatchNorm1d(128), nn.BatchNorm1d(1024)
        self.feat_tnet = TNet(k=64) if feature_transform else None

    def forward(self, x: torch.Tensor):
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(2, 1)
        t_in = self.input_tnet(x)  # (B, 3, 3)
        x = torch.bmm(t_in, x)  # align input
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)

        t_feat = None
        if self.feat_tnet is not None:
            t_feat = self.feat_tnet(x)  # (B, 64, 64)
            x = torch.bmm(t_feat, x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # (B, 1024, N)
        x = torch.max(x, dim=2)[0]  # (B, 1024) global feature
        return x, t_feat


class PointNetClassifier(nn.Module):
    def __init__(self, num_classes: int, feature_transform: bool = True):
        super().__init__()
        self.encoder = PointNetEncoder(feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1, self.bn2 = nn.BatchNorm1d(512), nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor):
        feat, t_feat = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(feat)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        return self.fc3(x), t_feat  # logits, feature matrix


def feature_transform_regularizer(t_feat: torch.Tensor) -> torch.Tensor:
    """Encourage the 64x64 feature transform to stay close to orthogonal.

    Keeps the learned transform from collapsing information; ||I - A Aᵀ||².
    """
    if t_feat is None:
        return torch.tensor(0.0)
    k = t_feat.size(1)
    identity = torch.eye(k, device=t_feat.device).unsqueeze(0)
    prod = torch.bmm(t_feat, t_feat.transpose(2, 1))
    return torch.mean(torch.norm(prod - identity, dim=(1, 2)))
