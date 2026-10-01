import numpy as np

from orion.config import PredictConfig
from orion.predict import KalmanPredictor, build_predictor
from orion.track import KalmanBox, Track


def _track(state):
    kf = KalmanBox((0, 0, 10, 10), 0.05, 0.6)
    kf.x[:] = np.array(state, dtype=np.float64)
    return Track(1, kf)


def test_kalman_predictor_linear_motion():
    tr = _track([100.0, 100.0, 10.0, 0.0, 10.0, 10.0, 0.0, 0.0])
    cfg = PredictConfig(horizon_s=0.3, dt=0.1)
    pred = KalmanPredictor(cfg).predict(tr)
    assert pred.shape == (3, 4)
    centers = (pred[:, 0] + pred[:, 2]) / 2
    # cx should advance 10 px per 0.1 s step
    assert np.allclose(centers, [101.0, 102.0, 103.0], atol=1e-3)


def test_predictor_foresees_looming():
    # growing box -> predicted widths must grow
    tr = _track([100.0, 100.0, 0.0, 0.0, 20.0, 20.0, 40.0, 40.0])
    cfg = PredictConfig(horizon_s=0.3, dt=0.1)
    pred = KalmanPredictor(cfg).predict(tr)
    widths = pred[:, 2] - pred[:, 0]
    assert widths[0] < widths[-1]


def test_build_predictor_defaults_to_kalman():
    cfg = PredictConfig(backend="kalman")
    assert isinstance(build_predictor(cfg), KalmanPredictor)
