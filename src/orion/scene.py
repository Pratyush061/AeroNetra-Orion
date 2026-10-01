"""Synthetic aerial scene generator.

Produces a deterministic, download-free stream of oblique aerial frames with
ground-truth boxes, per-object velocities and ego-motion. Two renders of the
same instant are produced:

* ``sharp`` — the "true" scene (what an event camera's edges correspond to);
* ``gray``  — the same scene with **injected motion blur**, standing in for a
  conventional rolling-shutter RGB frame during fast UAV flight.

That gap is exactly what the event-guided fusion stage exists to close, so the
demo is a faithful, honest illustration of the method rather than a toy.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import SceneConfig


@dataclass
class SceneFrame:
    """One rendered instant of the synthetic scene."""

    index: int
    time: float
    rgb: np.ndarray            # (H, W, 3) uint8
    gray: np.ndarray           # (H, W) uint8, motion-blurred
    sharp: np.ndarray          # (H, W) uint8, un-blurred
    boxes: np.ndarray          # (M, 4) float32 xyxy
    labels: list[str]
    velocities: np.ndarray     # (M, 2) float32 px/s
    ego_velocity: tuple[float, float]


def linear_motion_blur(gray: np.ndarray, angle_deg: float, length: int) -> np.ndarray:
    """Apply a linear motion-blur kernel of the given length and angle."""
    length = int(max(1, round(length)))
    if length <= 1:
        return gray
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    total = float(kernel.sum())
    if total > 0:
        kernel /= total
    return cv2.filter2D(gray, -1, kernel)


class SyntheticAerialScene:
    """A reproducible aerial scene: a scrolling ground plane with traffic."""

    def __init__(self, cfg: SceneConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._init_world()

    # ------------------------------------------------------------------ world
    def _init_world(self) -> None:
        cfg = self.cfg
        H, W = cfg.height, cfg.width

        # Low-texture, low-contrast ground so ego-motion produces few events and
        # few frame differences; the high-contrast vehicles then stand out.
        base = self.rng.integers(112, 138, size=(H, W), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (0, 0), 4.0)

        self.road_y0 = int(H * 0.55)
        self.road_y1 = int(H * 0.95)
        road = base.copy()
        band = road[self.road_y0 : self.road_y1, :].astype(np.int16)
        road[self.road_y0 : self.road_y1, :] = np.clip(band + 10, 0, 255).astype(np.uint8)
        self.background = road

        self.vehicles: list[dict] = []
        for _ in range(cfg.num_vehicles):
            w = int(self.rng.integers(26, 46))
            h = int(self.rng.integers(18, 30))
            x = int(self.rng.integers(0, W - w))
            y = int(self.rng.integers(self.road_y0, self.road_y1 - h))
            vx = float(self.rng.uniform(20, 70)) * (1 if self.rng.random() > 0.5 else -1)
            # high-contrast paint: either bright or dark against the mid-grey ground
            if self.rng.random() > 0.5:
                color = tuple(int(c) for c in self.rng.integers(205, 250, size=3))
            else:
                color = tuple(int(c) for c in self.rng.integers(25, 70, size=3))
            self.vehicles.append(
                dict(x=x, y=y, w=w, h=h, vx=vx, vy=0.0, color=color, label="vehicle")
            )

        self.intruders: list[dict] = []
        for _ in range(cfg.num_intruders):
            w0 = h0 = int(self.rng.integers(16, 24))
            # start near the top, roughly on the flight axis, so the descent
            # genuinely crosses the UAV's forward corridor
            x = int(W / 2 - w0 / 2 + self.rng.integers(-40, 41))
            y = int(self.rng.integers(0, int(H * 0.22)))
            self.intruders.append(
                dict(
                    x=x, y=y, w=w0, h=h0, w0=w0, h0=h0,
                    vx=float(self.rng.uniform(-15, 15)),
                    vy=float(self.rng.uniform(45, 70)),   # toward the camera
                    grow=float(self.rng.uniform(cfg.intruder_growth_min, cfg.intruder_growth_max)),
                    color=(30, 30, 240), label="intruder",
                )
            )

        self.bg_offset = 0.0

    # ----------------------------------------------------------------- render
    def _render(self, t: float):
        cfg = self.cfg
        W = cfg.width

        shift = int(round(self.bg_offset)) % W
        background = np.roll(self.background, shift, axis=1)
        rgb = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        boxes, labels, vels = [], [], []
        for ent in self.vehicles + self.intruders:
            x = int(round(ent["x"]))
            y = int(round(ent["y"]))
            w = int(ent["w"])
            h = int(ent["h"])
            cv2.rectangle(rgb, (x, y), (x + w, y + h), ent["color"], -1)
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (15, 15, 15), 1)
            boxes.append([x, y, x + w, y + h])
            labels.append(ent["label"])
            vels.append([ent["vx"], ent["vy"]])

        sharp = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        if cfg.motion_blur_px > 1:
            gray = linear_motion_blur(sharp, 0.0, cfg.motion_blur_px)
        else:
            gray = sharp

        return (
            rgb,
            gray,
            sharp,
            np.array(boxes, np.float32).reshape(-1, 4),
            labels,
            np.array(vels, np.float32).reshape(-1, 2),
        )

    def _step(self, dt: float) -> None:
        cfg = self.cfg
        W, H = cfg.width, cfg.height
        self.bg_offset += cfg.ego_speed
        for ent in self.vehicles:
            ent["x"] += ent["vx"] * dt
            if ent["x"] > W:
                ent["x"] = -ent["w"]
            elif ent["x"] + ent["w"] < 0:
                ent["x"] = W
        for ent in self.intruders:
            ent["x"] += ent["vx"] * dt
            ent["y"] += ent["vy"] * dt
            # perspective: an approaching intruder grows in apparent size
            ent["w"] *= (1.0 + ent["grow"] * dt)
            ent["h"] *= (1.0 + ent["grow"] * dt)
            if ent["y"] > H:
                ent["y"] = -ent["h0"]
                ent["w"], ent["h"] = ent["w0"], ent["h0"]
            if ent["x"] > W:
                ent["x"] = -ent["w"]
            elif ent["x"] + ent["w"] < 0:
                ent["x"] = W

    def __iter__(self):
        cfg = self.cfg
        dt = 1.0 / cfg.fps
        for i in range(cfg.frames):
            t = i * dt
            rgb, gray, sharp, boxes, labels, vels = self._render(t)
            yield SceneFrame(
                index=i, time=t, rgb=rgb, gray=gray, sharp=sharp,
                boxes=boxes, labels=labels, velocities=vels,
                ego_velocity=(cfg.ego_speed, 0.0),
            )
            self._step(dt)

    def __len__(self) -> int:
        return int(self.cfg.frames)
