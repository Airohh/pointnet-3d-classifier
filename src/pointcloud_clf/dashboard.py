"""Streamlit demo: upload a 3D mesh, see the predicted CAD category and the
sampled point cloud. Run: streamlit run src/pointcloud_clf/dashboard.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from pointcloud_clf import constants as C
from pointcloud_clf.data import load_pointcloud, normalize
from pointcloud_clf.predict import predict_array

st.set_page_config(page_title="3D CAD Classifier", layout="wide")
st.title("3D CAD Part Classifier — PointNet")

if not C.MODEL_PATH.exists():
    st.warning("No trained model found. Run `python scripts/train_model.py` first.")
    st.stop()

uploaded = st.file_uploader("Upload a mesh", type=["off", "ply", "stl", "obj"])
if uploaded is not None:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    points = normalize(load_pointcloud(tmp_path, C.NUM_POINTS))
    result = predict_array(points)

    left, right = st.columns(2)
    with left:
        st.subheader(f"Prediction: {result['prediction']}")
        st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        st.write("Top-3:")
        for r in result["top_k"]:
            st.write(f"- {r['label']}: {r['probability'] * 100:.1f}%")
    with right:
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode="markers", marker=dict(size=2))])
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    Path(tmp_path).unlink(missing_ok=True)
