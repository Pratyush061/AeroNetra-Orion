import numpy as np

from orion.events import (
    EventStream,
    event_rate,
    events_to_image,
    events_to_timesurface,
    simulate_events,
)


def _blink_sequence():
    H = W = 32
    f0 = np.zeros((H, W), np.uint8)
    f1 = np.zeros((H, W), np.uint8)
    f1[10:20, 10:20] = 255
    f2 = np.zeros((H, W), np.uint8)
    return [f0, f1, f2]


def test_simulate_events_basic():
    frames = _blink_sequence()
    ev = simulate_events(frames, [0.0, 0.1, 0.2], threshold=0.2)
    assert isinstance(ev, EventStream)
    assert len(ev) > 0
    assert set(np.unique(ev.p)).issubset({-1, 1})
    assert -1 in set(ev.p) and 1 in set(ev.p)          # both polarities
    assert np.all(np.diff(ev.t) >= 0)                   # time ordered
    assert ev.width == 32 and ev.height == 32


def test_simulate_events_requires_frames():
    import pytest

    with pytest.raises(ValueError):
        simulate_events([], [])


def test_events_to_image_normalised():
    ev = simulate_events(_blink_sequence(), [0.0, 0.1, 0.2], threshold=0.2)
    img = events_to_image(ev)
    assert img.shape == (32, 32, 2)
    assert img.dtype == np.float32
    assert img.max() <= 1.0 + 1e-6
    assert img[:, :, 0].sum() > 0 and img[:, :, 1].sum() > 0


def test_timesurface_range():
    ev = simulate_events(_blink_sequence(), [0.0, 0.1, 0.2], threshold=0.2)
    surf = events_to_timesurface(ev, t_now=0.2, tau=0.05)
    assert surf.shape == (32, 32)
    assert surf.min() >= 0.0 and surf.max() <= 1.0 + 1e-6


def test_event_rate():
    ev = simulate_events(_blink_sequence(), [0.0, 0.1, 0.2], threshold=0.2)
    assert event_rate(ev, 0.1) == len(ev) / 0.1


def test_empty_stream():
    ev = EventStream.empty_stream(10, 5)
    assert ev.empty()
    assert events_to_image(ev).shape == (5, 10, 2)
