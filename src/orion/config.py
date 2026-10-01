"""Typed configuration objects for AeroNetra-Orion.

Configuration is expressed as nested dataclasses so every field is discoverable,
type-hinted and serialisable, and can be overridden from a YAML file.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EventConfig:
    """Neuromorphic sensor (DVS) simulation parameters."""

    threshold: float = 0.18          # log-intensity contrast threshold (C)
    window_ms: float = 15.0          # accumulation window for event frames
    contrast_eps: float = 1.0        # epsilon for log() to avoid log(0)


@dataclass
class SceneConfig:
    """Synthetic aerial scene generator parameters (download-free demo)."""

    width: int = 640
    height: int = 480
    num_vehicles: int = 6
    num_intruders: int = 1
    frames: int = 72
    fps: float = 30.0
    ego_speed: float = 2.5           # px/frame horizontal camera translation
    motion_blur_px: int = 7          # injected blur length (0 disables)
    intruder_growth_min: float = 0.5  # apparent-size growth/s for approaching intruders
    intruder_growth_max: float = 0.9
    seed: int = 7


@dataclass
class FusionConfig:
    """Event-guided fusion parameters."""

    sharpen_strength: float = 0.7
    event_gain: float = 0.5
    temporal_alpha: float = 0.35
    use_events: bool = True


@dataclass
class DetectConfig:
    """Detection parameters (classical by default, optional ONNX)."""

    background_alpha: float = 0.05
    diff_threshold: int = 20
    min_area: int = 40
    max_area_frac: float = 0.25
    morph_kernel: int = 3
    onnx_path: str | None = None
    score_threshold: float = 0.35


@dataclass
class TrackConfig:
    """Multi-object tracker parameters."""

    iou_threshold: float = 0.15
    max_age: int = 10
    min_hits: int = 2
    process_noise: float = 0.05
    measurement_noise: float = 0.6


@dataclass
class PredictConfig:
    """Predictive world-model parameters."""

    horizon_s: float = 1.0
    dt: float = 1.0 / 30.0
    backend: str = "kalman"          # "kalman" or "torch"
    torch_model: str | None = None
    safety_radius_px: float = 46.0
    # forward keep-out corridor, as fractions of width/height
    corridor_x0: float = 0.44
    corridor_y0: float = 0.38
    corridor_x1: float = 0.56
    corridor_y1: float = 0.62


@dataclass
class OrionConfig:
    """Top-level configuration."""

    events: EventConfig = field(default_factory=EventConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    output_dir: str = "outputs"
    save_video: bool = True
    save_frames: bool = False
    dataset: str | None = None
    data_root: str = "data"


def _build(cls: type, data: dict[str, Any] | None):
    data = data or {}
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        target = hints[f.name]
        value = data[f.name]
        if dataclasses.is_dataclass(target) and isinstance(value, dict):
            value = _build(target, value)
        kwargs[f.name] = value
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> OrionConfig:
    """Build an :class:`OrionConfig` from a plain dictionary."""
    return _build(OrionConfig, data)


def load_config(path: str | Path | None = None) -> OrionConfig:
    """Load configuration from YAML, or return defaults when ``path`` is None."""
    if path is None:
        return OrionConfig()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping, got {type(data).__name__}")
    return config_from_dict(data)


def to_dict(cfg: OrionConfig) -> dict[str, Any]:
    """Serialise an :class:`OrionConfig` to a nested dictionary."""
    return dataclasses.asdict(cfg)
