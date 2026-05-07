"""High-level wrapper around the Ultralytics YOLO API.

Exposes a small, typed surface (`YOLODetector`, `Detection`) that the rest of
the project (CLI, Streamlit demo, ONNX exporter, tests) consumes uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class Detection:
    """A single detected object in a frame.

    Coordinates are in the original image's pixel space (`xyxy`).
    `track_id` is populated only when running in tracking mode.
    """

    box_xyxy: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None


class YOLODetector:
    """Thin wrapper over `ultralytics.YOLO` for predict + track + export."""

    def __init__(
        self,
        model: str = "yolo12n.pt",
        device: Union[str, int, None] = "auto",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ) -> None:
        # Imported lazily so unit tests that only touch dataclasses / drawing
        # helpers don't need PyTorch installed.
        from ultralytics import YOLO

        self.model_path = model
        self.model = YOLO(model)
        self.device = None if device == "auto" else device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.class_names = self._normalize_class_names(self.model.names)

    @staticmethod
    def _normalize_class_names(names) -> dict:
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        return {i: n for i, n in enumerate(names)}

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Run a single forward pass and return parsed detections."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._parse(results)

    def track(self, frame: np.ndarray, persist: bool = True) -> List[Detection]:
        """Run inference + ByteTrack to keep stable IDs across frames."""
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            persist=persist,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        return self._parse(results)

    def export_onnx(
        self,
        dynamic: bool = True,
        half: bool = False,
        imgsz: Optional[int] = None,
        opset: int = 12,
    ) -> Path:
        """Export the underlying model to ONNX and return the output path."""
        out = self.model.export(
            format="onnx",
            dynamic=dynamic,
            half=half,
            imgsz=imgsz or self.imgsz,
            opset=opset,
        )
        return Path(out)

    def _parse(self, results) -> List[Detection]:
        detections: List[Detection] = []
        if not results:
            return detections
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        ids = (
            r.boxes.id.cpu().numpy().astype(int)
            if r.boxes.id is not None
            else [None] * len(boxes)
        )

        for box, conf, cls_id, tid in zip(boxes, confs, cls_ids, ids):
            x1, y1, x2, y2 = (int(v) for v in box)
            detections.append(
                Detection(
                    box_xyxy=(x1, y1, x2, y2),
                    class_id=int(cls_id),
                    class_name=self.class_names.get(int(cls_id), str(int(cls_id))),
                    confidence=float(conf),
                    track_id=int(tid) if tid is not None else None,
                )
            )
        return detections
