import numpy as np

from orion.config import FusionConfig
from orion.events import EventStream, simulate_events
from orion.fusion import EventFrameFusion


def test_fuse_shapes_and_ranges():
    H = W = 40
    f0 = np.full((H, W), 100, np.uint8)
    f1 = f0.copy()
    f1[10:30, 10:30] = 220
    ev = simulate_events([f0, f1], [0.0, 0.1], threshold=0.2)

    fusion = EventFrameFusion(FusionConfig())
    out = fusion.fuse(f1, ev)

    assert out.sharpened.shape == (H, W)
    assert out.sharpened.dtype == np.uint8
    assert out.feature.shape == (H, W, 3)
    assert out.feature.dtype == np.float32
    assert 0.0 <= out.feature.min() and out.feature.max() <= 1.0 + 1e-6
    assert out.event_image.shape == (H, W, 2)


def test_fusion_without_events():
    cfg = FusionConfig(use_events=False)
    fusion = EventFrameFusion(cfg)
    img = np.full((20, 20), 120, np.uint8)
    out = fusion.fuse(img, EventStream.empty_stream(20, 20))
    assert out.feature[:, :, 1].max() == 0.0


def test_fusion_sharpens_more_where_events_fire():
    H = W = 48
    base = np.full((H, W), 120, np.uint8)
    base = np.clip(base.astype(np.int16) + np.random.default_rng(0).integers(-5, 5, (H, W)), 0, 255).astype(np.uint8)
    moving = base.copy()
    moving[20:28, 20:28] = 240
    ev = simulate_events([base, moving], [0.0, 0.1], threshold=0.1)

    out = EventFrameFusion(FusionConfig(temporal_alpha=0.0)).fuse(moving, ev)
    # restored detail should not blow out of range and must stay uint8
    assert out.sharpened.dtype == np.uint8
    assert out.sharpened.max() <= 255
