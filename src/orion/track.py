"""Multi-object tracking: greedy IoU association with per-track Kalman filters.

Each track keeps a constant-velocity Kalman filter over the state
``[cx, cy, vx, vy, w, h]``. Because the predictor downstream reuses this state,
the tracker and the world model share one consistent motion representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import TrackConfig
from .detect import Detection


def iou(a, b) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def box_from_state(state: np.ndarray) -> tuple[float, float, float, float]:
    """Convert ``[cx, cy, vx, vy, w, h]`` to an xyxy box."""
    cx, cy, w, h = float(state[0]), float(state[1]), float(state[4]), float(state[5])
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class KalmanBox:
    """Constant-velocity Kalman filter over a bounding box.

    State is ``[cx, cy, vx, vy, w, h, vw, vh]``. Tracking *size velocity* is
    what lets the predictor foresee looming and therefore estimate TTC.
    """

    def __init__(self, box, process_noise: float, measurement_noise: float):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = x2 - x1, y2 - y1
        self.x = np.array([cx, cy, 0.0, 0.0, w, h, 0.0, 0.0], np.float64)
        self.F = np.eye(8)
        self.H = np.zeros((4, 8))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 4] = self.H[3, 5] = 1.0
        self.P = np.eye(8) * 10.0
        self.Q = np.eye(8) * process_noise
        self.R = np.eye(4) * measurement_noise

    def predict(self, dt: float = 1.0) -> np.ndarray:
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.F[4, 6] = dt
        self.F[5, 7] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, box) -> np.ndarray:
        x1, y1, x2, y2 = box
        z = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.x.copy()

    @property
    def box(self) -> tuple[float, float, float, float]:
        return box_from_state(self.x)


@dataclass
class Track:
    """A single tracked object."""

    track_id: int
    kf: KalmanBox
    label: str = "object"
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    history: list = field(default_factory=list)

    @property
    def state(self) -> np.ndarray:
        return self.kf.x.copy()

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.kf.box


class IoUKalmanTracker:
    """Greedy-IoU association tracker with Kalman smoothing."""

    def __init__(self, cfg: TrackConfig):
        self.cfg = cfg
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        for tr in self.tracks:
            tr.kf.predict(1.0)
            tr.age += 1
            tr.time_since_update += 1

        matched_tr: set[int] = set()
        matched_det: set[int] = set()

        if self.tracks and detections:
            cost = np.zeros((len(self.tracks), len(detections)), np.float64)
            for i, tr in enumerate(self.tracks):
                for j, d in enumerate(detections):
                    cost[i, j] = iou(tr.box, d.box)
            pairs = [
                (cost[i, j], i, j)
                for i in range(len(self.tracks))
                for j in range(len(detections))
                if cost[i, j] >= self.cfg.iou_threshold
            ]
            pairs.sort(reverse=True)
            for _, i, j in pairs:
                if i in matched_tr or j in matched_det:
                    continue
                matched_tr.add(i)
                matched_det.add(j)
                tr = self.tracks[i]
                tr.kf.update(detections[j].box)
                tr.hits += 1
                tr.time_since_update = 0
                tr.history.append(tr.box)

        for j, d in enumerate(detections):
            if j in matched_det:
                continue
            kf = KalmanBox(d.box, self.cfg.process_noise, self.cfg.measurement_noise)
            tr = Track(self._next_id, kf, label=d.label)
            tr.history.append(tr.box)
            self.tracks.append(tr)
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.cfg.max_age]
        return [
            t
            for t in self.tracks
            if t.hits >= self.cfg.min_hits and t.time_since_update == 0
        ]
