FROM python:3.12-slim

# libGL + glib needed by trimesh for some mesh formats
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image small
RUN pip install --no-cache-dir torch>=2.2 --index-url https://download.pytorch.org/whl/cpu
# Runtime-only deps (no mlflow/streamlit/sklearn/onnx/pytest) — the serving
# image stays lean; training/analysis tooling lives in requirements.txt.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY models/ ./models/
ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "pointcloud_clf.api:app", "--host", "0.0.0.0", "--port", "8000"]
