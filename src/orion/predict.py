"""Predictive world model.

The core bet of AeroNetra-Orion is *anticipation*: instead of reacting to where
objects are, forecast where they will be. Two interchangeable predictors share
one contract — given a track, return the next ``T`` predicted boxes:

* :class:`KalmanPredictor` — propagates the tracker's constant-velocity Kalman
  state forward. Deterministic, weight-free, always available.
* :class:`TorchWorldModel` — a small GRU that learns residual dynamics from a
  window of past boxes. Optional; falls back cleanly when PyTorch is absent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PredictConfig
from .track import Track, box_from_state


def _steps(horizon_s: float, dt: float) -> int:
    return max(1, int(round(horizon_s / dt)))


class KalmanPredictor:
    """Constant-velocity forecast from the tracker's Kalman state."""

    def __init__(self, cfg: PredictConfig):
        self.cfg = cfg

    def predict(
        self, track: Track, horizon_s: float | None = None, dt: float | None = None
    ) -> np.ndarray:
        horizon_s = self.cfg.horizon_s if horizon_s is None else horizon_s
        dt = self.cfg.dt if dt is None else dt
        n = _steps(horizon_s, dt)

        F = np.eye(8)
        F[0, 2] = dt
        F[1, 3] = dt
        F[4, 6] = dt
        F[5, 7] = dt
        cur = track.state
        boxes = np.empty((n, 4), np.float32)
        for i in range(n):
            cur = F @ cur
            boxes[i] = box_from_state(cur)
        return boxes


@dataclass
class TorchWorldModel:
    """GRU world model predicting future box offsets from a past window."""

    model_path: str | None = None
    past: int = 5
    future: int = 30
    hidden: int = 64

    def __post_init__(self):
        import torch

        self._torch = torch
        self.net = self._build()
        if self.model_path:
            state = torch.load(self.model_path, map_location="cpu")
            self.net.load_state_dict(state)
        self.net.eval()

    def _build(self):
        import torch.nn as nn

        future, hidden = self.future, self.hidden

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(4, hidden, batch_first=True)
                self.head = nn.Linear(hidden, future * 4)

            def forward(self, x):
                h, _ = self.gru(x)
                out = self.head(h[:, -1])
                return out.view(-1, future, 4)

        return _Net()

    def predict(self, track: Track, horizon_s=None, dt=None) -> np.ndarray:
        torch = self._torch
        hist = np.asarray(track.history[-self.past :], np.float32)
        if len(hist) < self.past:
            pad = np.repeat(hist[:1], self.past - len(hist), axis=0)
            hist = np.vstack([pad, hist])
        # normalise relative to the last observed box
        ref = hist[-1]
        cx, cy = (ref[0] + ref[2]) / 2.0, (ref[1] + ref[3]) / 2.0
        w = max(1e-3, ref[2] - ref[0])
        h = max(1e-3, ref[3] - ref[1])
        norm = hist.copy()
        norm[:, [0, 2]] = (norm[:, [0, 2]] - cx) / w
        norm[:, [1, 3]] = (norm[:, [1, 3]] - cy) / h
        with torch.no_grad():
            inp = torch.from_numpy(norm[None, :, :])
            out = self.net(inp)[0].numpy()
        boxes = np.empty((self.future, 4), np.float32)
        boxes[:, [0, 2]] = out[:, [0, 2]] * w + cx
        boxes[:, [1, 3]] = out[:, [1, 3]] * h + cy
        return boxes


def build_predictor(cfg: PredictConfig):
    """Return the configured predictor, falling back to Kalman if torch is absent."""
    if cfg.backend == "torch":
        try:
            return TorchWorldModel(model_path=cfg.torch_model, future=_steps(cfg.horizon_s, cfg.dt))
        except ImportError:
            pass
    return KalmanPredictor(cfg)
