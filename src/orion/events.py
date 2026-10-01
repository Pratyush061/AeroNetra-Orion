"""Neuromorphic event simulation and event representations.

Implements the standard DVS (Dynamic Vision Sensor) model: a pixel emits an
event when the logarithmic intensity change since that pixel's last event
exceeds a contrast threshold ``C``. This is how real event cameras (DAVIS,
Prophesee, the sensors behind MVSEC / DSEC) produce asynchronous,
high-dynamic-range, low-latency data that is immune to motion blur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EventStream:
    """An asynchronous stream of polarity events."""

    x: np.ndarray      # (N,) int32 column
    y: np.ndarray      # (N,) int32 row
    t: np.ndarray      # (N,) float64 seconds
    p: np.ndarray      # (N,) int8  +1 (brighten) / -1 (darken)
    width: int
    height: int

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def empty(self) -> bool:
        return len(self) == 0

    @staticmethod
    def empty_stream(width: int, height: int) -> "EventStream":
        return EventStream(
            np.zeros(0, np.int32),
            np.zeros(0, np.int32),
            np.zeros(0, np.float64),
            np.zeros(0, np.int8),
            int(width),
            int(height),
        )


def log_intensity(frame: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Return the log-intensity image ``log(I + eps)`` as float32."""
    return np.log(np.asarray(frame, dtype=np.float32) + eps)


def simulate_events(
    frames,
    timestamps,
    threshold: float = 0.18,
    eps: float = 1.0,
) -> EventStream:
    """Convert a sequence of grayscale frames into a DVS event stream.

    Parameters
    ----------
    frames:
        Iterable of 2-D grayscale images (uint8 or float).
    timestamps:
        Monotonic timestamps (seconds) aligned with ``frames``.
    threshold:
        Contrast threshold ``C`` on the log-intensity difference.
    eps:
        Epsilon used inside ``log`` to keep it finite.

    Returns
    -------
    EventStream
        Events ordered by emission (which is frame order here).
    """
    frames = [np.asarray(f) for f in frames]
    if len(frames) == 0:
        raise ValueError("simulate_events requires at least one frame")
    if len(frames) != len(timestamps):
        raise ValueError("frames and timestamps must have equal length")

    H, W = frames[0].shape[:2]
    ref = log_intensity(frames[0], eps)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    ps: list[np.ndarray] = []

    for i in range(1, len(frames)):
        t = float(timestamps[i])
        L = log_intensity(frames[i], eps)
        delta = L - ref
        mask = np.abs(delta) >= threshold
        if mask.any():
            yy, xx = np.nonzero(mask)
            xs.append(xx.astype(np.int32))
            ys.append(yy.astype(np.int32))
            ts.append(np.full(xx.shape, t, dtype=np.float64))
            ps.append(np.sign(delta[mask]).astype(np.int8))
            ref = np.where(mask, L, ref)

    if xs:
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        t = np.concatenate(ts)
        p = np.concatenate(ps)
    else:
        return EventStream.empty_stream(W, H)
    return EventStream(x, y, t, p, W, H)


def events_to_image(
    events: EventStream,
    width: int | None = None,
    height: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Accumulate events into a 2-channel image ``[positive, negative]``.

    Returns a float32 array of shape ``(H, W, 2)``. When ``normalize`` is set
    the channels are scaled to ``[0, 1]`` by the global maximum count.
    """
    W = int(width or events.width)
    H = int(height or events.height)
    img = np.zeros((H, W, 2), np.float32)
    if len(events):
        pos = events.p > 0
        neg = ~pos
        np.add.at(img[:, :, 0], (events.y[pos], events.x[pos]), 1.0)
        np.add.at(img[:, :, 1], (events.y[neg], events.x[neg]), 1.0)
    if normalize:
        m = float(img.max())
        if m > 0:
            img /= m
    return img


def events_to_timesurface(
    events: EventStream,
    t_now: float,
    width: int | None = None,
    height: int | None = None,
    tau: float = 0.02,
) -> np.ndarray:
    """Surface of Active Events: ``exp(-(t_now - t_last) / tau)`` per pixel."""
    W = int(width or events.width)
    H = int(height or events.height)
    last = np.full((H, W), -np.inf, np.float64)
    if len(events):
        order = np.argsort(events.t)  # oldest -> newest, newest wins on ties
        last[events.y[order], events.x[order]] = events.t[order]
    surface = np.zeros((H, W), np.float32)
    valid = np.isfinite(last)
    surface[valid] = np.exp(-np.maximum(0.0, t_now - last[valid]) / tau)
    return surface


def event_rate(events: EventStream, dt: float) -> float:
    """Average events per second over a window of duration ``dt``."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    return len(events) / dt
