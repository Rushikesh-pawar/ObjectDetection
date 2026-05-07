"""Export an Ultralytics YOLO model to ONNX for deployment.

Examples
--------
    python export_onnx.py --model yolo12n.pt --imgsz 640 --dynamic
    python export_onnx.py --model yolo12s.pt --half
"""

from __future__ import annotations

import argparse
import sys

from src.detector import YOLODetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export YOLOv12 weights to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="yolo12n.pt", help="Source weights (.pt)")
    p.add_argument("--imgsz", type=int, default=640, help="Static export image size")
    p.add_argument("--dynamic", action="store_true", help="Use dynamic axes")
    p.add_argument("--half", action="store_true", help="Export FP16 (GPU only)")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    detector = YOLODetector(model=args.model, imgsz=args.imgsz)
    out = detector.export_onnx(
        dynamic=args.dynamic, half=args.half, imgsz=args.imgsz, opset=args.opset
    )
    print(f"Exported: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
