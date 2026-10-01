"""Event-guided fusion.

A conventional frame captured during fast UAV flight is motion-blurred; an
event camera observing the *same* scene does not blur. We exploit that
asymmetry: the asynchronous event stream marks where moving edges are, and we
use that activity to modulate a high-frequency restoration of the blurred
frame. The result is a sharper intensity image plus a fused feature tensor that
downstream detection consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import FusionConfig
from .events import EventStream, events_to_image


@dataclass
class FusedFrame:
    """Output of the fusion stage."""

    sharpened: np.ndarray     # (H, W) uint8
    feature: np.ndarray       # (H, W, 3) float32 in [0, 1]
    event_image: np.ndarray   # (H, W, 2) float32 in [0, 1]


class EventFrameFusion:
    """Fuse a (blurred) intensity frame with an event stream."""

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None

    def fuse(
        self,
        gray: np.ndarray,
        events: EventStream,
        width: int | None = None,
        height: int | None = None,
    ) -> FusedFrame:
        cfg = self.cfg
        g = np.asarray(gray, dtype=np.float32)
        H, W = g.shape[:2]

        # 1. Base high-frequency restoration (unsharp mask).
        low = cv2.GaussianBlur(g, (0, 0), 1.2)
        high = g - low

        # 2. Event activity map (where the scene is actually changing).
        if cfg.use_events:
            ev = events_to_image(events, W, H)
        else:
            ev = np.zeros((H, W, 2), np.float32)
        activity = np.clip(ev[:, :, 0] + ev[:, :, 1], 0.0, 1.0)

        # 3. Event-modulated sharpening: restore harder where events fire.
        boost = 1.0 + cfg.event_gain * activity
        restored = g + cfg.sharpen_strength * boost * high

        # 4. Temporal consistency across frames.
        if self._prev is not None and self._prev.shape == restored.shape:
            restored = (1.0 - cfg.temporal_alpha) * restored + cfg.temporal_alpha * self._prev

        sharpened = np.clip(restored, 0, 255).astype(np.uint8)
        self._prev = sharpened.astype(np.float32)

        feature = np.dstack(
            [sharpened.astype(np.float32) / 255.0, ev[:, :, 0], ev[:, :, 1]]
        ).astype(np.float32)
        return FusedFrame(sharpened=sharpened, feature=feature, event_image=ev)
