import numpy as np

from orion.config import TrackConfig
from orion.detect import Detection
from orion.track import IoUKalmanTracker, KalmanBox, Track, box_from_state, iou


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0.0 < iou((0, 0, 10, 10), (5, 5, 15, 15)) < 1.0


def test_box_from_state_roundtrip():
    state = np.array([50.0, 60.0, 0, 0, 20.0, 10.0, 0, 0])
    assert box_from_state(state) == (40.0, 55.0, 60.0, 65.0)


def test_kalman_constant_velocity():
    kf = KalmanBox((0, 0, 10, 10), 0.05, 0.6)
    kf.x[:] = np.array([100.0, 100.0, 10.0, 0.0, 10.0, 10.0, 0.0, 0.0])
    kf.predict(1.0)
    assert abs(kf.x[0] - 110.0) < 1e-6


def test_tracker_keeps_identity_on_moving_object():
    tracker = IoUKalmanTracker(TrackConfig(min_hits=1))
    ids = []
    for i in range(6):
        box = (10 + i * 5, 10, 30 + i * 5, 30)
        confirmed = tracker.update([Detection(box, 0.9)])
        if confirmed:
            ids.append(confirmed[0].track_id)
    assert len(ids) >= 3
    assert len(set(ids)) == 1  # one stable identity


def test_tracker_drops_stale_tracks():
    tracker = IoUKalmanTracker(TrackConfig(max_age=1, min_hits=1))
    tracker.update([Detection((0, 0, 10, 10), 0.9)])
    tracker.update([])
    tracker.update([])
    assert tracker.tracks == []


def test_track_state_shape():
    tr = Track(1, KalmanBox((0, 0, 10, 10), 0.05, 0.6))
    assert tr.state.shape == (8,)
    assert len(tr.box) == 4
