"""Run a YOLOv12 ONNX model with onnxruntime alone (no PyTorch required).

This demonstrates the deployment path: after `export_onnx.py`, the model can
run on any ONNX Runtime provider (CPU, CUDA, CoreML, DirectML, OpenVINO).

Examples
--------
    python onnx_infer.py --onnx yolo12n.onnx --image samples/dog.jpg
    python onnx_infer.py --onnx yolo12n.onnx --image bus.jpg --out result.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES_FILE = Path(__file__).parent / "coco.names"


def _load_class_names() -> List[str]:
    return CLASS_NAMES_FILE.read_text().strip().splitlines()


def _letterbox(img: np.ndarray, imgsz: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Letterbox-resize an image to a square `imgsz` canvas (preserves aspect)."""
    h0, w0 = img.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_x = (imgsz - nw) // 2
    pad_y = (imgsz - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, r, (pad_x, pad_y)


def _preprocess(img: np.ndarray, imgsz: int):
    canvas, ratio, pad = _letterbox(img, imgsz)
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, normalize
    blob = blob.transpose(2, 0, 1)[None]  # HWC -> NCHW
    return np.ascontiguousarray(blob), ratio, pad


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    idxs = scores.argsort()[::-1]
    keep: List[int] = []
    while idxs.size > 0:
        i = int(idxs[0])
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_rest - inter + 1e-7)
        idxs = rest[iou < iou_thr]
    return keep


def _postprocess(
    out: np.ndarray,
    ratio: float,
    pad: Tuple[int, int],
    conf_thr: float,
    iou_thr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ultralytics ONNX output shape: (1, 4 + num_classes, num_boxes)."""
    pred = out[0].T  # (num_boxes, 4 + num_classes)
    cxcywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    cls_ids = cls_scores.argmax(1)
    confs = cls_scores.max(1)
    mask = confs > conf_thr
    cxcywh, confs, cls_ids = cxcywh[mask], confs[mask], cls_ids[mask]
    if len(cxcywh) == 0:
        return np.empty((0, 4), int), np.empty((0,), float), np.empty((0,), int)

    pad_x, pad_y = pad
    x1 = (cxcywh[:, 0] - cxcywh[:, 2] / 2 - pad_x) / ratio
    y1 = (cxcywh[:, 1] - cxcywh[:, 3] / 2 - pad_y) / ratio
    x2 = (cxcywh[:, 0] + cxcywh[:, 2] / 2 - pad_x) / ratio
    y2 = (cxcywh[:, 1] + cxcywh[:, 3] / 2 - pad_y) / ratio
    boxes = np.stack([x1, y1, x2, y2], 1)
    keep = _nms(boxes, confs, iou_thr)
    return boxes[keep].astype(int), confs[keep], cls_ids[keep]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a YOLO ONNX model with onnxruntime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--onnx", required=True, help="Path to exported .onnx model")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--out", default="onnx_output.jpg", help="Annotated output path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    providers = ort.get_available_providers()
    sess = ort.InferenceSession(args.onnx, providers=providers)
    in_name = sess.get_inputs()[0].name

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: could not read image '{args.image}'", file=sys.stderr)
        return 1

    blob, ratio, pad = _preprocess(img, args.imgsz)
    t0 = time.perf_counter()
    out = sess.run(None, {in_name: blob})[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    boxes, confs, cls_ids = _postprocess(out, ratio, pad, args.conf, args.iou)

    names = _load_class_names()
    for box, conf, cid in zip(boxes, confs, cls_ids):
        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{names[int(cid)] if cid < len(names) else cid} {conf*100:.0f}%"
        cv2.putText(
            img, label, (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA,
        )

    cv2.imwrite(args.out, img)
    print(
        f"Detected {len(boxes)} objects in {latency_ms:.1f} ms "
        f"(provider: {sess.get_providers()[0]}). Wrote {args.out}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
