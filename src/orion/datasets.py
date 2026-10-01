"""Dataset adapters.

AeroNetra-Orion runs with **zero downloads** (see the synthetic scene), but it
also speaks to the datasets that actually matter for UAV and event-based
perception. No single dataset is privileged: event datasets (DSEC, MVSEC,
VisEvent, DAVIS) and aerial frame datasets (AU-AIR, UAVDT, SeaDronesSee,
DroneCrowd, DOTA, CARPK) are all first-class, and VisDrone is simply one option
among many.

Adapters expose small, pure parsing functions (easy to unit-test) plus loaders
that materialise the project's :class:`~orion.scene.SceneFrame` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .events import EventStream


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for a supported dataset."""

    key: str
    name: str
    kind: str        # "event" | "aerial"
    modality: str
    url: str
    note: str


_REGISTRY: dict[str, DatasetSpec] = {
    # ---- event / neuromorphic -------------------------------------------
    "dsec": DatasetSpec(
        "dsec", "DSEC", "event", "stereo event + frame",
        "https://dsec.ifi.uzh.ch/",
        "High-res stereo event+frame driving data; ideal for event-guided fusion and depth.",
    ),
    "mvsec": DatasetSpec(
        "mvsec", "MVSEC", "event", "event + frame + IMU + LiDAR",
        "https://daniilidis-group.github.io/mvsec/",
        "Multi-sensor event data; strong for event-based optical flow and odometry.",
    ),
    "visevent": DatasetSpec(
        "visevent", "VisEvent", "event", "event + frame tracking",
        "https://sites.google.com/view/viseventtrack/",
        "Event-based single-object tracking; directly exercises the predictive tracker.",
    ),
    "davis240c": DatasetSpec(
        "davis240c", "DAVIS 240C", "event", "event + frame",
        "https://rpg.ifi.uzh.ch/davis_data.html",
        "Classic DAVIS recordings for event representation and deblurring baselines.",
    ),
    # ---- aerial / UAV frame ---------------------------------------------
    "auair": DatasetSpec(
        "auair", "AU-AIR", "aerial", "aerial RGB",
        "https://bozcani.github.io/auairdataset",
        "Low-altitude UAV traffic imagery with boxes; good UAV-domain detector training.",
    ),
    "uavdt": DatasetSpec(
        "uavdt", "UAVDT", "aerial", "aerial RGB video",
        "https://sites.google.com/view/uavdt-dataset",
        "UAV detection & tracking benchmark; feeds both detection and tracking stages.",
    ),
    "seadronessee": DatasetSpec(
        "seadronessee", "SeaDronesSee", "aerial", "maritime UAV RGB",
        "https://seadronessee.cs.uni-tuebingen.de/",
        "Maritime search-and-rescue UAV data; stresses small-object detection.",
    ),
    "dronecrowd": DatasetSpec(
        "dronecrowd", "DroneCrowd", "aerial", "UAV crowd counting",
        "https://github.com/VisDrone/DroneCrowd",
        "Dense crowd/vehicle counting from UAV; exercises predictive occupancy.",
    ),
    "dota": DatasetSpec(
        "dota", "DOTA", "aerial", "aerial oriented boxes",
        "https://captain-whu.github.io/DOTA/",
        "Large aerial oriented-object benchmark; pairs with rotated detection work.",
    ),
    "carpk": DatasetSpec(
        "carpk", "CARPK", "aerial", "aerial parking counting",
        "https://lafi.github.io/LPN/",
        "Car-parking counting from UAV; a clean counting benchmark.",
    ),
    "visdrone": DatasetSpec(
        "visdrone", "VisDrone", "aerial", "aerial RGB detection/tracking",
        "https://github.com/VisDrone/VisDrone-Dataset",
        "Widely used aerial detection/tracking benchmark — one option among many.",
    ),
}


def list_datasets() -> list[DatasetSpec]:
    """Return all supported datasets."""
    return list(_REGISTRY.values())


def get_dataset(key: str) -> DatasetSpec:
    """Return the spec for ``key`` or raise ``KeyError``."""
    if key not in _REGISTRY:
        raise KeyError(f"Unknown dataset {key!r}. Known: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[key]


def dataset_dir(key: str, root: str | Path = "data") -> Path:
    """Return the expected local directory for a dataset."""
    return Path(root) / key


def is_available(key: str, root: str | Path = "data") -> bool:
    """True if the dataset directory exists and is non-empty."""
    d = dataset_dir(key, root)
    return d.is_dir() and any(d.iterdir())


def require(key: str, root: str | Path = "data") -> Path:
    """Return the dataset directory or raise an actionable error."""
    spec = get_dataset(key)
    d = dataset_dir(key, root)
    if not is_available(key, root):
        raise FileNotFoundError(
            f"Dataset '{spec.name}' not found at {d}.\n"
            f"  Download: {spec.url}\n"
            f"  Then place the extracted files under {d}.\n"
            f"  ({spec.note})"
        )
    return d


# ---------------------------------------------------------------- parsing ---
def parse_visdrone_det(lines) -> list[tuple[float, float, float, float]]:
    """Parse VisDrone DET annotation lines into xyxy boxes.

    Format: ``bbox_left,bbox_top,bbox_width,bbox_height,score,category,...``
    Rows with category ``0`` (ignored regions) are skipped.
    """
    boxes = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        x, y, w, h = (float(parts[i]) for i in range(4))
        category = int(float(parts[5]))
        if category == 0:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def parse_uavdt_gt(lines) -> list[tuple[float, float, float, float]]:
    """Parse UAVDT ground-truth lines into xyxy boxes.

    Format: ``frame,object_id,x,y,w,h,out_of_view,occlusion,...``
    """
    boxes = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        x, y, w, h = (float(parts[i]) for i in range(2, 6))
        boxes.append((x, y, x + w, y + h))
    return boxes


def parse_auair(json_obj: dict) -> list[tuple[float, float, float, float]]:
    """Parse an AU-AIR annotation JSON object into xyxy boxes."""
    boxes = []
    for ann in json_obj.get("annotations", []):
        b = ann.get("bbox")
        if not b:
            continue
        x, y, w, h = b["left"], b["top"], b["width"], b["height"]
        boxes.append((float(x), float(y), float(x + w), float(y + h)))
    return boxes


# ----------------------------------------------------------------- loading ---
def load_events_npz(path: str | Path) -> EventStream:
    """Load an event stream from an ``.npz`` with keys ``x, y, t, p`` (+ ``width, height``)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Event file not found: {path}")
    data = np.load(path)
    required = {"x", "y", "t", "p"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Event npz missing keys {sorted(missing)}; found {sorted(data.files)}")
    x = np.asarray(data["x"], np.int32)
    y = np.asarray(data["y"], np.int32)
    t = np.asarray(data["t"], np.float64)
    p = np.asarray(data["p"], np.int8)
    width = int(data["width"]) if "width" in data.files else int(x.max()) + 1
    height = int(data["height"]) if "height" in data.files else int(y.max()) + 1
    return EventStream(x, y, t, p, width, height)
