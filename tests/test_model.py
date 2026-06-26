import torch

from pointcloud_clf.model import (
    PointNetClassifier,
    TNet,
    feature_transform_regularizer,
)


def test_classifier_output_shape():
    model = PointNetClassifier(num_classes=10).eval()
    x = torch.randn(4, 1024, 3)
    logits, t_feat = model(x)
    assert logits.shape == (4, 10)
    assert t_feat.shape == (4, 64, 64)


def test_tnet_returns_square_matrix():
    net = TNet(k=3).eval()
    out = net(torch.randn(2, 3, 512))
    assert out.shape == (2, 3, 3)


def test_permutation_invariance():
    """Max-pool aggregator => same logits regardless of point order."""
    model = PointNetClassifier(num_classes=10).eval()
    x = torch.randn(1, 1024, 3)
    perm = torch.randperm(1024)
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(x[:, perm, :])
    assert torch.allclose(a, b, atol=1e-4)


def test_feature_transform_regularizer_zero_for_identity():
    identity = torch.eye(64).unsqueeze(0).repeat(3, 1, 1)
    assert feature_transform_regularizer(identity).item() < 1e-5
