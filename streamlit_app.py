"""Streamlit web demo for the YOLOv12 detector.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.detector import YOLODetector
from src.visualize import draw_detections

st.set_page_config(
    page_title="YOLOv12 Object Detection",
    page_icon="🎯",
    layout="wide",
)

st.title("YOLOv12 — Real-Time Object Detection")
st.caption(
    "Upload an image or video and run the latest YOLO model in your browser. "
    "Built with Ultralytics + Streamlit."
)

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Model",
        ["yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt"],
        index=0,
        help="Larger models are slower but more accurate.",
    )
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)
    track = st.checkbox("Enable tracking (videos)", value=True)
    st.markdown("---")
    st.caption("First run downloads weights (≈6 MB for `yolo12n`).")


@st.cache_resource(show_spinner="Loading model…")
def _load_detector(model: str, conf: float, iou: float) -> YOLODetector:
    return YOLODetector(model=model, conf=conf, iou=iou)


detector = _load_detector(model_choice, conf, iou)
detector.conf = conf
detector.iou = iou

tab_image, tab_video = st.tabs(["🖼️  Image", "🎞️  Video"])


with tab_image:
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        rgb = np.array(pil_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        t0 = time.perf_counter()
        detections = detector.predict(bgr)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        annotated_bgr = draw_detections(bgr.copy(), detections)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Original")
            st.image(rgb, use_container_width=True)
        with col_b:
            st.subheader("Detections")
            st.image(annotated_rgb, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Objects detected", len(detections))
        m2.metric("Inference latency", f"{latency_ms:.1f} ms")
        m3.metric("Throughput", f"{1000.0 / max(latency_ms, 1e-6):.1f} FPS")

        if detections:
            st.subheader("Per-class summary")
            counts = Counter(d.class_name for d in detections)
            st.bar_chart(dict(counts))

            st.subheader("Detections")
            st.dataframe(
                [
                    {
                        "class": d.class_name,
                        "confidence": round(d.confidence, 3),
                        "x1": d.box_xyxy[0],
                        "y1": d.box_xyxy[1],
                        "x2": d.box_xyxy[2],
                        "y2": d.box_xyxy[3],
                    }
                    for d in detections
                ],
                use_container_width=True,
            )


with tab_video:
    uploaded_v = st.file_uploader(
        "Upload a video", type=["mp4", "mov", "avi", "mkv"], key="video"
    )
    if uploaded_v is not None:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded_v.name).suffix
        ) as tmp:
            tmp.write(uploaded_v.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            st.error("Could not open the uploaded video.")
        else:
            placeholder = st.empty()
            progress = st.progress(0.0)
            stats = st.empty()
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            i = 0
            t_start = time.perf_counter()
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    detections = (
                        detector.track(frame) if track else detector.predict(frame)
                    )
                    annotated = draw_detections(frame, detections)
                    placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )
                    i += 1
                    progress.progress(min(i / n_frames, 1.0))
                    elapsed = time.perf_counter() - t_start
                    stats.caption(
                        f"Frame {i}/{n_frames}  •  "
                        f"{i / max(elapsed, 1e-6):.1f} avg FPS  •  "
                        f"{len(detections)} objects in last frame"
                    )
            finally:
                cap.release()
