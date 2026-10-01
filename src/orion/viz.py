"""Visualisation helpers (OpenCV drawing only, no GUI dependency)."""

from __future__ import annotations

import cv2
import numpy as np

from .detect import Detection
from .track import Track
from .ttc import RiskAssessment

RISK_COLORS = {
    "critical": (0, 0, 255),
    "warning": (0, 165, 255),
    "nominal": (0, 200, 0),
}


def draw_detections(img: np.ndarray, detections: list[Detection], color=(200, 200, 200)) -> None:
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)


def draw_tracks(img: np.ndarray, tracks: list[Track]) -> None:
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(img, f"#{t.track_id}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)


def draw_predictions(img: np.ndarray, assessments: list[RiskAssessment]) -> None:
    for a in assessments:
        color = RISK_COLORS.get(a.risk, (0, 200, 0))
        centers = [
            (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)) for b in a.predicted
        ]
        for p, q in zip(centers[:-1], centers[1:]):
            cv2.line(img, p, q, color, 1, cv2.LINE_AA)
        if centers:
            x1, y1, x2, y2 = (int(v) for v in a.predicted[-1])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)


def draw_hud(
    img: np.ndarray,
    n_det: int,
    n_trk: int,
    mttc: float,
    alerts: int,
    event_rate: float | None = None,
) -> None:
    lines = [
        f"detections: {n_det}",
        f"tracks: {n_trk}",
        "min TTC: " + ("inf" if not np.isfinite(mttc) else f"{mttc:.2f}s"),
        f"critical alerts: {alerts}",
    ]
    if event_rate is not None:
        lines.append(f"events/s: {event_rate:,.0f}")
    y = 22
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 20


def overlay_occupancy(img: np.ndarray, occ: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    if occ is None or occ.max() <= 0:
        return img
    heat = cv2.applyColorMap((occ * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = occ > 0.02
    img[mask] = cv2.addWeighted(img[mask], 1.0 - alpha, heat[mask], alpha, 0)
    return img


def annotate(
    rgb: np.ndarray,
    detections: list[Detection],
    tracks: list[Track],
    assessments: list[RiskAssessment],
    occupancy: np.ndarray | None = None,
    mttc: float = float("inf"),
    alerts: int = 0,
    event_rate: float | None = None,
) -> np.ndarray:
    """Return an annotated copy of an RGB frame."""
    img = np.ascontiguousarray(rgb.copy())
    if occupancy is not None:
        overlay_occupancy(img, occupancy)
    draw_detections(img, detections)
    draw_tracks(img, tracks)
    draw_predictions(img, assessments)
    draw_hud(img, len(detections), len(tracks), mttc, alerts, event_rate)
    return img
