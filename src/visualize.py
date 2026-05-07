"""Drawing helpers for annotated frames."""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

from .detector import Detection

_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (56, 211, 0),     # green
    (235, 119, 52),   # orange
    (235, 52, 198),   # magenta
    (52, 168, 235),   # cyan
    (245, 218, 66),   # yellow
    (149, 52, 235),   # purple
    (52, 235, 162),   # mint
    (235, 52, 52),    # red
)


def _color_for(class_id: int) -> Tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def draw_detections(frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Draw bounding boxes + labels for each detection. Mutates and returns `frame`."""
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        color = _color_for(det.class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.confidence * 100:.0f}%"
        if det.track_id is not None:
            label = f"#{det.track_id} {label}"

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        bg_y1 = max(0, y1 - th - baseline - 4)
        cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_hud(frame: np.ndarray, fps: float, n_objects: int) -> np.ndarray:
    """Draw an FPS / object-count overlay in the top-left corner."""
    text = f"FPS: {fps:5.1f}   Objects: {n_objects}"
    cv2.rectangle(frame, (8, 8), (260, 38), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame
