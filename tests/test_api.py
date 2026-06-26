import io

import trimesh
from fastapi.testclient import TestClient

from pointcloud_clf.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_classes():
    r = client.get("/classes")
    assert r.status_code == 200
    assert len(r.json()["classes"]) == 10


def test_predict_rejects_bad_extension():
    r = client.post("/predict", files={"file": ("bad.txt", b"nope", "text/plain")})
    assert r.status_code == 400


def test_predict_path_without_model_or_with_model():
    """A valid mesh either returns a prediction (model trained) or a clean 503."""
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    buf = io.BytesIO()
    mesh.export(buf, file_type="ply")
    buf.seek(0)
    r = client.post("/predict", files={"file": ("box.ply", buf.read(), "application/octet-stream")})
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert "prediction" in body and "top_k" in body
