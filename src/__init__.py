"""Real-time object detection package built on YOLOv12 (Ultralytics)."""

from .detector import Detection, YOLODetector
from .visualize import draw_detections, draw_hud

__all__ = ["Detection", "YOLODetector", "draw_detections", "draw_hud"]
__version__ = "1.0.0"
