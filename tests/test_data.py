import numpy as np
import trimesh

from pointcloud_clf.data import augment, normalize, sample_mesh


def test_sample_mesh_shape():
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    pts = sample_mesh(mesh, n_points=512)
    assert pts.shape == (512, 3)
    assert pts.dtype == np.float32


def test_normalize_unit_sphere():
    pts = np.random.RandomState(0).randn(1000, 3).astype(np.float32) * 10 + 5
    out = normalize(pts)
    # centered
    assert np.allclose(out.mean(axis=0), 0, atol=1e-5)
    # within unit sphere
    assert np.max(np.linalg.norm(out, axis=1)) <= 1.0 + 1e-5


def test_augment_preserves_shape():
    rng = np.random.default_rng(0)
    pts = normalize(np.random.RandomState(1).randn(256, 3).astype(np.float32))
    out = augment(pts, rng)
    assert out.shape == pts.shape
    # rotation+jitter must actually change the cloud
    assert not np.allclose(out, pts)
