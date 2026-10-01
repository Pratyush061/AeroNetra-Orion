"""Object detection.

Two interchangeable detectors share one output contract:

* :class:`MotionDetector` — a classical, weight-free detector that runs the
  moment the repo is cloned. It is a background-subtraction detector over the
  fused frame, which is exactly the regime event cameras help with.
* :class:`OnnxDetector` — an optional learned detector that consumes an
  exported YOLOv8 ONNX model through OpenCV DNN, for higher-quality boxes.

Keeping the output contract identical means the tracker, predictor and
collision modules never need to know which detector produced a box.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetectConfig


@dataclass
class Detection:
    """A single detection in xyxy pixel coordinates."""

    box: tuple[float, float, float, float]
    score: float
    label: str = "object"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


class MotionDetector:
    """Background-subtraction detector over a grayscale frame."""

    def __init__(self, cfg: DetectConfig):
        self.cfg = cfg
        self._bg: np.ndarray | None = None
        k = max(1, int(cfg.morph_kernel))
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def reset(self) -> None:
        self._bg = None

    def detect(self, gray: np.ndarray) -> list[Detection]:
        g = np.asarray(gray, dtype=np.float32)
        if self._bg is None:
            self._bg = g.copy()
            return []

        self._bg = (1.0 - self.cfg.background_alpha) * self._bg + self.cfg.background_alpha * g
        diff = np.abs(g - self._bg)
        mask = (diff >= self.cfg.diff_threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        H, W = g.shape[:2]
        max_area = self.cfg.max_area_frac * H * W

        detections: list[Detection] = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.cfg.min_area or area > max_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            score = float(min(1.0, area / (self.cfg.min_area * 8.0)))
            detections.append(Detection((x, y, x + w, y + h), score, "object"))
        return detections


class OnnxDetector:
    """Optional YOLOv8-style ONNX detector via OpenCV DNN."""

    def __init__(self, cfg: DetectConfig, input_size: int = 640):
        if not cfg.onnx_path:
            raise ValueError("DetectConfig.onnx_path must be set to use OnnxDetector")
        import os

        if not os.path.exists(cfg.onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {cfg.onnx_path}")
        self.cfg = cfg
        self.input_size = input_size
        self.net = cv2.dnn.readNetFromONNX(cfg.onnx_path)

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        H, W = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            bgr, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        out = self.net.forward()
        preds = self._parse(out, W, H)
        return preds

    def _parse(self, out: np.ndarray, W: int, H: int) -> list[Detection]:
        # YOLOv8 export: (1, 4 + num_classes, num_anchors)
        if out.ndim != 3:
            raise ValueError(f"Unexpected ONNX output shape {out.shape}")
        if out.shape[1] < out.shape[2]:
            preds = out[0].T  # (num_anchors, 4 + C)
        else:
            preds = out[0]
        boxes = preds[:, :4]
        scores = preds[:, 4:]
        cls = scores.argmax(axis=1)
        conf = scores[np.arange(len(scores)), cls]
        keep = conf >= self.cfg.score_threshold
        sx, sy = W / self.input_size, H / self.input_size
        detections: list[Detection] = []
        for b, c, s in zip(boxes[keep], cls[keep], conf[keep]):
            cx, cy, bw, bh = b
            x1 = (cx - bw / 2) * sx
            y1 = (cy - bh / 2) * sy
            x2 = (cx + bw / 2) * sx
            y2 = (cy + bh / 2) * sy
            detections.append(Detection((x1, y1, x2, y2), float(s), f"cls{int(c)}"))
        return detections


def build_detector(cfg: DetectConfig):
    """Return an ONNX detector when a model is configured, else the motion detector."""
    if cfg.onnx_path:
        return OnnxDetector(cfg)
    return MotionDetector(cfg)
