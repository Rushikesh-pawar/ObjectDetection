"""Smoke tests for the benchmark utility — uses a fake detector to avoid loading weights."""

from __future__ import annotations

import time
from typing import List

import numpy as np

from src.benchmark import BenchmarkResult, run_benchmark


class _FakeDetector:
    """Mimics the surface of YOLODetector that the benchmark calls."""

    def __init__(self) -> None:
        self.model_path = "fake.pt"
        self.device = "cpu"

    def predict(self, frame: np.ndarray) -> List[object]:
        time.sleep(0.001)  # simulate small latency
        return []


def test_run_benchmark_smoke() -> None:
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    result = run_benchmark(_FakeDetector(), img, iterations=5, warmup=2)

    assert isinstance(result, BenchmarkResult)
    assert result.iterations == 5
    assert result.imgsz == 640
    assert result.avg_latency_ms > 0
    assert result.fps > 0
    # P50 should be <= P95 by definition
    assert result.p50_latency_ms <= result.p95_latency_ms


def test_benchmark_as_dict_roundtrip() -> None:
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    result = run_benchmark(_FakeDetector(), img, iterations=3, warmup=0)
    payload = result.as_dict()
    assert {"model", "device", "imgsz", "iterations", "fps"} <= payload.keys()
