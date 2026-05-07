"""Real-time object detection CLI.

Examples
--------
    # Webcam (default)
    python app.py --source 0

    # Video file with tracking, save annotated MP4
    python app.py --source video.mp4 --track --save out.mp4

    # Image
    python app.py --source samples/dog.jpg --save annotated.jpg

    # Run on the bigger model on GPU
    python app.py --source video.mp4 --model yolo12m.pt --device 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

from src.detector import YOLODetector
from src.visualize import draw_detections, draw_hud

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time object detection with YOLOv12.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default="0", help="Webcam index, image, or video path")
    p.add_argument("--model", default="yolo12n.pt", help="Ultralytics YOLO weights")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps | <index>")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--track", action="store_true", help="Enable ByteTrack tracking")
    p.add_argument("--save", default=None, help="Path to save annotated output")
    p.add_argument("--no-display", action="store_true", help="Run headless")
    return p.parse_args()


def _is_image_path(source: str) -> bool:
    return Path(source).suffix.lower() in IMAGE_EXTS


def _run_image(detector: YOLODetector, path: str, save: Optional[str], display: bool) -> int:
    img = cv2.imread(path)
    if img is None:
        print(f"Error: could not read image '{path}'", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    detections = detector.predict(img)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Detected {len(detections)} objects in {latency_ms:.1f} ms")
    img = draw_detections(img, detections)
    img = draw_hud(img, 1000.0 / max(latency_ms, 1e-6), len(detections))
    if save:
        cv2.imwrite(save, img)
        print(f"Saved annotated image to {save}")
    if display:
        cv2.imshow("YOLOv12 Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def _run_stream(
    detector: YOLODetector,
    source: str,
    track: bool,
    save: Optional[str],
    display: bool,
) -> int:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        print(f"Error: could not open source '{source}'", file=sys.stderr)
        return 1

    writer: Optional[cv2.VideoWriter] = None
    fps_ema: float = 0.0
    frame_count = 0
    t_start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()
            detections = detector.track(frame) if track else detector.predict(frame)
            inst_fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
            fps_ema = inst_fps if frame_count == 0 else 0.9 * fps_ema + 0.1 * inst_fps

            frame = draw_detections(frame, detections)
            frame = draw_hud(frame, fps_ema, len(detections))

            if save and writer is None:
                h, w = frame.shape[:2]
                src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                writer = cv2.VideoWriter(
                    save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h)
                )
            if writer is not None:
                writer.write(frame)

            if display:
                cv2.imshow("YOLOv12 Detection", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            frame_count += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    print(
        f"Processed {frame_count} frames in {elapsed:.1f}s "
        f"({frame_count / max(elapsed, 1e-6):.1f} avg FPS)"
    )
    if save:
        print(f"Annotated output saved to {save}")
    return 0


def main() -> int:
    args = parse_args()
    detector = YOLODetector(
        model=args.model,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    display = not args.no_display
    if not args.source.isdigit() and _is_image_path(args.source):
        return _run_image(detector, args.source, args.save, display)
    return _run_stream(detector, args.source, args.track, args.save, display)


if __name__ == "__main__":
    sys.exit(main())
