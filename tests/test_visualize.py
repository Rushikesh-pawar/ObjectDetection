"""Drawing helpers don't need a model loaded — they just operate on numpy arrays."""

from __future__ import annotations

import numpy as np

from src.detector import Detection
from src.visualize import draw_detections, draw_hud


def test_detection_dataclass_defaults() -> None:
    d = Detection(box_xyxy=(10, 20, 30, 40), class_id=0, class_name="person", confidence=0.9)
    assert d.track_id is None
    assert d.box_xyxy == (10, 20, 30, 40)
    assert d.class_name == "person"


def test_draw_detections_returns_same_shape() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    dets = [
        Detection(box_xyxy=(10, 20, 60, 80), class_id=0, class_name="person", confidence=0.9),
        Detection(box_xyxy=(70, 30, 140, 90), class_id=2, class_name="car", confidence=0.8, track_id=7),
    ]
    out = draw_detections(frame, dets)
    assert out.shape == frame.shape
    # Drawing should leave non-zero pixels somewhere
    assert out.sum() > 0


def test_draw_detections_handles_empty_list() -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    out = draw_detections(frame, [])
    assert out.shape == frame.shape
    assert out.sum() == 0


def test_draw_hud_renders_overlay() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out = draw_hud(frame, fps=30.0, n_objects=5)
    assert out.shape == frame.shape
    assert out.sum() > 0
