"""Time-to-collision and predictive occupancy.

Forecast trajectories are converted into the two things a UAV flight controller
actually needs:

* a per-object **time-to-collision** — the predicted time until the object's
  trajectory enters the UAV's forward keep-out corridor; and
* a **predictive occupancy** map — where the scene will be busy over the next
  horizon, so the platform can avoid it before it is there.

The corridor is the central band of the image (where a forward-flying UAV is
headed); entering it, or being predicted to enter it within the horizon, is the
collision signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import PredictConfig
from .track import Track


@dataclass
class RiskAssessment:
    """Collision assessment for one track."""

    track_id: int
    ttc_s: float
    min_dist_px: float
    risk: str                    # "critical" | "warning" | "nominal"
    predicted: np.ndarray        # (T, 4) forecast boxes


def corridor_rect(cfg: PredictConfig, width: int, height: int):
    """Forward keep-out rectangle in pixel coordinates."""
    return (
        cfg.corridor_x0 * width,
        cfg.corridor_y0 * height,
        cfg.corridor_x1 * width,
        cfg.corridor_y1 * height,
    )


def _overlaps(box, rect) -> bool:
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = rect
    ix1, iy1 = max(x1, rx1), max(y1, ry1)
    ix2, iy2 = min(x2, rx2), min(y2, ry2)
    return ix1 < ix2 and iy1 < iy2


def assess_track(
    track: Track,
    predicted: np.ndarray,
    cfg: PredictConfig,
    frame_shape: tuple[int, int],
) -> RiskAssessment:
    """Assess one track against the forward keep-out corridor."""
    H, W = frame_shape[:2]
    rect = corridor_rect(cfg, W, H)
    cx0, cy0 = W / 2.0, H / 2.0

    ttc = float("inf")
    min_dist = float("inf")
    for i, b in enumerate(predicted):
        if _overlaps(b, rect):
            ttc = min(ttc, (i + 1) * cfg.dt)
        ccx, ccy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        min_dist = min(min_dist, math.hypot(ccx - cx0, ccy - cy0))

    if ttc <= cfg.horizon_s:
        risk = "critical"
    elif ttc <= 2.0 * cfg.horizon_s:
        risk = "warning"
    else:
        risk = "nominal"

    return RiskAssessment(track.track_id, ttc, min_dist, risk, predicted)


def assess_tracks(
    tracks: list[Track],
    predictions: list[np.ndarray],
    cfg: PredictConfig,
    frame_shape: tuple[int, int],
) -> list[RiskAssessment]:
    """Assess a batch of tracks."""
    return [assess_track(t, p, cfg, frame_shape) for t, p in zip(tracks, predictions)]


def occupancy_map(
    assessments: list[RiskAssessment],
    width: int,
    height: int,
    step_index: int = -1,
    sigma: float = 12.0,
) -> np.ndarray:
    """Gaussian predictive-occupancy map at a given future step."""
    occ = np.zeros((height, width), np.float32)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    two_sigma2 = 2.0 * sigma * sigma
    for a in assessments:
        if len(a.predicted) == 0:
            continue
        idx = min(max(step_index if step_index >= 0 else len(a.predicted) + step_index, 0),
                  len(a.predicted) - 1)
        b = a.predicted[idx]
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        x0 = max(0, int(cx - 3 * sigma))
        x1 = min(width, int(cx + 3 * sigma) + 1)
        y0 = max(0, int(cy - 3 * sigma))
        y1 = min(height, int(cy + 3 * sigma) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / two_sigma2)
        occ[y0:y1, x0:x1] = np.maximum(occ[y0:y1, x0:x1], g)
    return occ


def min_ttc(assessments: list[RiskAssessment]) -> float:
    """Smallest finite TTC across all assessments (inf if none)."""
    vals = [a.ttc_s for a in assessments if np.isfinite(a.ttc_s)]
    return float(min(vals)) if vals else float("inf")
