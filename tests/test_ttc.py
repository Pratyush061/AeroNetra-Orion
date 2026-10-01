import numpy as np

from orion.config import PredictConfig
from orion.track import KalmanBox, Track
from orion.ttc import assess_track, assess_tracks, corridor_rect, min_ttc, occupancy_map

H, W = 480, 640


def _track(box):
    return Track(1, KalmanBox(box, 0.05, 0.6))


def test_corridor_rect():
    cfg = PredictConfig()
    x0, y0, x1, y1 = corridor_rect(cfg, W, H)
    assert x0 < x1 and y0 < y1


def test_risk_critical_when_predicted_into_corridor():
    cfg = PredictConfig(horizon_s=1.0, dt=0.1)
    box = (W / 2 - 10, H / 2 - 10, W / 2 + 10, H / 2 + 10)
    pred = np.tile(np.array(box, np.float32), (5, 1))
    a = assess_track(_track(box), pred, cfg, (H, W))
    assert a.risk == "critical"
    assert a.ttc_s <= cfg.horizon_s


def test_risk_nominal_when_away():
    cfg = PredictConfig(horizon_s=0.5, dt=0.1)
    box = (2, 2, 12, 12)
    pred = np.tile(np.array(box, np.float32), (5, 1))
    a = assess_track(_track(box), pred, cfg, (H, W))
    assert a.risk == "nominal"
    assert a.ttc_s == float("inf")


def test_min_ttc_and_batch():
    cfg = PredictConfig(horizon_s=1.0, dt=0.1)
    box = (W / 2 - 5, H / 2 - 5, W / 2 + 5, H / 2 + 5)
    pred = np.tile(np.array(box, np.float32), (4, 1))
    out = assess_tracks([_track(box)], [pred], cfg, (H, W))
    assert len(out) == 1
    assert min_ttc(out) <= 1.0


def test_occupancy_map_shape():
    cfg = PredictConfig()
    box = (W / 2 - 10, H / 2 - 10, W / 2 + 10, H / 2 + 10)
    pred = np.tile(np.array(box, np.float32), (4, 1))
    a = assess_track(_track(box), pred, cfg, (H, W))
    occ = occupancy_map([a], W, H)
    assert occ.shape == (H, W)
    assert occ.max() > 0.0 and occ.min() >= 0.0
