"""FastAPI service: upload a 3D mesh, get the predicted CAD category.

Mirrors how the model would be exposed to a software application in
production: a single backend endpoint, model loaded once at startup, robust
to bad input.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from . import constants as C
from .predict import predict_file

app = FastAPI(
    title="PointCloud Classifier",
    version="0.1.0",
    description="Classify 3D CAD meshes with PointNet.",
)

ALLOWED = {".off", ".ply", ".stl", ".obj"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_present": C.MODEL_PATH.exists()}


@app.get("/classes")
def classes() -> dict:
    return {"classes": list(C.CLASSES)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"unsupported file type '{suffix}'; allowed: {sorted(ALLOWED)}")
    if not C.MODEL_PATH.exists():
        raise HTTPException(503, "model not trained yet; run scripts/train_model.py")

    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        result = predict_file(tmp_path)
    except Exception as exc:
        raise HTTPException(422, f"could not parse mesh: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"filename": file.filename, **result}
