# Real-Time Object Detection with YOLOv12

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv12-00C7B7.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.17%2B-005CED.svg)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end object detection demo built on top of **YOLOv12** (Ultralytics, 2025).
Runs in real time on webcam, video, or images; tracks objects across frames with
**ByteTrack**; exports to **ONNX** for deployment; ships with an interactive
**Streamlit** web demo and a reproducible latency benchmark.

> Originally a 70-line YOLOv3 + OpenCV-DNN script. Fully rewritten as a modular,
> typed Python project to demonstrate modern computer-vision engineering.

---

## Features

- **YOLOv12 inference** via the [Ultralytics](https://github.com/ultralytics/ultralytics) API — supports `n` / `s` / `m` / `l` / `x` variants
- **Multiple input modes** — webcam, image, video file (`--source`)
- **Multi-object tracking** — ByteTrack with persistent IDs (`--track`)
- **ONNX export + pure-onnxruntime inference path** — deploy without PyTorch (CPU / CUDA / CoreML / DirectML)
- **Latency benchmark CLI** — avg / P50 / P95 latency and FPS, JSON-friendly output
- **Streamlit web demo** — drag-and-drop image/video, live FPS, per-class breakdown
- **Modular `src/` package**, type hints throughout, pytest unit tests, MIT licensed

## Demo

```bash
# Webcam
python app.py --source 0

# Video with tracking, save annotated MP4
python app.py --source video.mp4 --track --save out.mp4

# Browser demo
streamlit run streamlit_app.py
```

---

## Project structure

```
ObjectDetection/
├── app.py                  # CLI: webcam / image / video, tracking, save output
├── streamlit_app.py        # Browser demo (image + video tabs)
├── benchmark_cli.py        # Latency / FPS benchmark
├── export_onnx.py          # Export YOLOv12 → ONNX
├── onnx_infer.py           # Pure-onnxruntime inference (no PyTorch needed)
├── src/
│   ├── detector.py         # YOLODetector class + Detection dataclass
│   ├── visualize.py        # Box / HUD drawing helpers
│   └── benchmark.py        # Benchmark routine (avg / P50 / P95 latency)
├── tests/                  # Pytest unit tests (no GPU required)
├── coco.names              # 80-class COCO label list
├── requirements.txt
├── LICENSE                 # MIT
└── README.md
```

## Quick start

```bash
git clone https://github.com/<your-username>/ObjectDetection.git
cd ObjectDetection

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Webcam (downloads YOLOv12n weights on first run, ~6 MB)
python app.py --source 0
```

> The first run downloads the requested weights from the Ultralytics asset hub.
> Use `--model yolo12s.pt` (or `m` / `l` / `x`) for higher accuracy at the cost of speed.

## Usage

### `app.py` — real-time CLI

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index, image path, or video path |
| `--model` | `yolo12n.pt` | Any Ultralytics-compatible weights (`.pt`) |
| `--device` | `auto` | `cpu`, `cuda`, `mps`, or device index |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--imgsz` | `640` | Inference image size |
| `--track` | off | Enable ByteTrack tracking (videos / streams) |
| `--save` | — | Path to save annotated output (image or MP4) |
| `--no-display` | off | Headless mode (for servers / Docker / CI) |

Press `q` or `Esc` to quit the live window.

### `benchmark_cli.py` — latency / throughput benchmark

```bash
python benchmark_cli.py --model yolo12n.pt --iters 200 --imgsz 640
python benchmark_cli.py --model yolo12s.pt --device cpu --json
```

Sample output:

```
Model:        yolo12n.pt
Device:       mps
Image size:   640x640
Iterations:   100
Avg latency:    18.26 ms
P50 latency:    18.04 ms
P95 latency:    19.61 ms
FPS:            54.8
```

### Measured results

Benchmarked on a **MacBook Air (Apple M4, 10-core, 16 GB RAM)**, 640×640 input, 100 timed iterations after 10 warmup runs:

| Model     | Device   | Avg latency | P50 latency | P95 latency | FPS  |
|-----------|----------|-------------|-------------|-------------|------|
| yolo12n   | MPS      |  18.26 ms   |  18.04 ms   |  19.61 ms   | 54.8 |
| yolo12s   | MPS      |  25.72 ms   |  25.65 ms   |  26.19 ms   | 38.9 |
| yolo12m   | MPS      |  58.73 ms   |  58.62 ms   |  59.57 ms   | 17.0 |
| yolo12n   | CPU      |  73.77 ms   |  73.59 ms   |  76.83 ms   | 13.6 |

Reproduce on your hardware:

```bash
python benchmark_cli.py --model yolo12n.pt --device mps --iters 100
python benchmark_cli.py --model yolo12s.pt --device mps --iters 100
python benchmark_cli.py --model yolo12m.pt --device mps --iters 100
python benchmark_cli.py --model yolo12n.pt --device cpu --iters 100
```

### ONNX export + deployment

```bash
# Export
python export_onnx.py --model yolo12n.pt --imgsz 640 --dynamic

# Run with onnxruntime (no PyTorch required)
python onnx_infer.py --onnx yolo12n.onnx --image samples/dog.jpg --out result.jpg
```

`onnx_infer.py` uses only `onnxruntime` + `opencv` + `numpy` — useful for
edge / serverless deployment where shipping the full PyTorch stack is too heavy.
ONNX Runtime auto-selects from available providers (CPU, CUDA, CoreML, DirectML,
OpenVINO, …).

### Tracking with ByteTrack

```bash
python app.py --source video.mp4 --track
```

Each detection is rendered with a stable integer ID (e.g. `#7 person 92%`) that
persists across frames, even through brief occlusions.

### Streamlit web demo

```bash
streamlit run streamlit_app.py
```

- Image tab — drag-and-drop, side-by-side original vs. detections, per-class bar
  chart, full detection table.
- Video tab — frame-by-frame preview with running FPS / object counts.

---

## Architecture

```
                ┌──────────────────┐
   frame  ───►  │  YOLODetector    │  ───►  list[Detection]
                │  (Ultralytics)   │
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │  ByteTrack       │   (--track flag)
                │  persistent IDs  │
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │  draw_detections │  ───►  annotated frame
                │  + draw_hud      │
                └──────────────────┘
```

The `Detection` dataclass is the single source of truth — every consumer (CLI,
Streamlit, ONNX runner, tests) operates on the same typed structure, which keeps
new outputs (JSON, CSV, MQTT, …) a one-function add.

## What was upgraded from the original

| Aspect            | Before (YOLOv3 / OpenCV-DNN, 2022)        | After (YOLOv12 / Ultralytics, 2026)                    |
|-------------------|--------------------------------------------|---------------------------------------------------------|
| Model             | YOLOv3 (2018), `.cfg` + `.weights` (~240 MB) | YOLOv12 (2025), single `.pt` (~6 MB for `n` variant)    |
| Architecture      | Darknet-53 CNN                             | Attention-centric (area attention, R-ELAN backbone)     |
| Code              | 70-line single script                      | Modular `src/` package, type hints, dataclasses, tests  |
| Inputs            | Webcam only                                | Webcam, image, video, headless                          |
| Tracking          | —                                          | ByteTrack persistent IDs                                |
| Deployment        | —                                          | ONNX export + onnxruntime inference path                |
| Demo              | —                                          | Streamlit web UI                                        |
| Reproducibility   | —                                          | Benchmark CLI, pinned `requirements.txt`, MIT license   |
| Config            | Hard-coded constants                       | `argparse` CLI flags everywhere                         |

## Tests

```bash
pip install pytest
pytest -q
```

The unit tests don't require GPU or model weights — they exercise the
`Detection` dataclass, the drawing helpers, and the benchmark utility against a
fake detector, so they run in a few seconds in CI.

## Roadmap

- [ ] Custom-dataset fine-tuning script (`train.py`)
- [ ] Dockerfile for one-command deployment
- [ ] TensorRT export path for NVIDIA Jetson
- [ ] Multi-camera RTSP support

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8 / 11 / 12 implementation and weights
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — multi-object tracking
- [ONNX Runtime](https://onnxruntime.ai/) — cross-platform inference

## License

[MIT](LICENSE) — free to use, modify, and distribute.

---

**Author:** Rushikesh Pawar — MS @ Northeastern University.
Open to feedback, issues, and PRs.
