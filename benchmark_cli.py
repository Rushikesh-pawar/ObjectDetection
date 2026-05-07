"""Benchmark inference latency / throughput for any Ultralytics-compatible model.

Examples
--------
    python benchmark_cli.py --model yolo12n.pt --iters 200 --imgsz 640
    python benchmark_cli.py --model yolo12s.pt --device cpu --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import cv2
import numpy as np

from src.benchmark import run_benchmark
from src.detector import YOLODetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark YOLO inference latency and throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="yolo12n.pt", help="Ultralytics YOLO weights")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps | <index>")
    p.add_argument("--image", default=None, help="Image path. Defaults to a synthetic test image.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--iters", type=int, default=100, help="Timed iterations")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    p.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON")
    return p.parse_args()


def _load_image(path: Optional[str], imgsz: int) -> np.ndarray:
    if path is None:
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def main() -> int:
    args = parse_args()
    img = _load_image(args.image, args.imgsz)
    detector = YOLODetector(model=args.model, device=args.device, imgsz=args.imgsz)
    result = run_benchmark(detector, img, iterations=args.iters, warmup=args.warmup)

    if args.as_json:
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    print(f"Model:        {result.model}")
    print(f"Device:       {result.device}")
    print(f"Image size:   {result.imgsz}x{result.imgsz}")
    print(f"Iterations:   {result.iterations}")
    print(f"Avg latency:  {result.avg_latency_ms:7.2f} ms")
    print(f"P50 latency:  {result.p50_latency_ms:7.2f} ms")
    print(f"P95 latency:  {result.p95_latency_ms:7.2f} ms")
    print(f"FPS:          {result.fps:7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
