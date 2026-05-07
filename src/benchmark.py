"""Latency / FPS benchmark utility."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

from .detector import YOLODetector


@dataclass
class BenchmarkResult:
    model: str
    device: str
    imgsz: int
    iterations: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    fps: float

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def run_benchmark(
    detector: YOLODetector,
    image: np.ndarray,
    iterations: int = 100,
    warmup: int = 10,
) -> BenchmarkResult:
    """Measure inference latency on a fixed image. Excludes pre/post-processing-free
    cases by going through the full predict() path the rest of the app uses."""
    for _ in range(max(0, warmup)):
        detector.predict(image)

    timings_ms = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        t0 = time.perf_counter()
        detector.predict(image)
        timings_ms[i] = (time.perf_counter() - t0) * 1000.0

    avg = float(timings_ms.mean())
    return BenchmarkResult(
        model=detector.model_path,
        device=str(detector.device or "auto"),
        imgsz=int(image.shape[1]),
        iterations=iterations,
        avg_latency_ms=avg,
        p50_latency_ms=float(np.percentile(timings_ms, 50)),
        p95_latency_ms=float(np.percentile(timings_ms, 95)),
        fps=(1000.0 / avg) if avg > 0 else 0.0,
    )
