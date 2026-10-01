"""Scene-generator robustness across resolutions (regression tests)."""

import numpy as np
import pytest

from orion.config import OrionConfig, SceneConfig
from orion.pipeline import run_demo
from orion.scene import SyntheticAerialScene


@pytest.mark.parametrize("w,h", [(48, 36), (96, 72), (160, 120), (640, 480), (800, 200)])
def test_scene_generates_at_any_resolution(w, h):
    cfg = SceneConfig(width=w, height=h, frames=4)
    scene = SyntheticAerialScene(cfg)
    frame = next(iter(scene))
    assert frame.gray.shape == (h, w)
    assert frame.boxes.shape[1] == 4
    # every ground-truth box must lie inside the frame
    for x1, y1, x2, y2 in frame.boxes:
        assert 0 <= x1 <= x2 <= w
        assert 0 <= y1 <= y2 <= h


def test_tiny_frames_run_end_to_end(tmp_path):
    cfg = OrionConfig()
    cfg.scene.width = 48
    cfg.scene.height = 36
    cfg.scene.frames = 4
    cfg.save_video = False
    summary = run_demo(cfg, out_dir=tmp_path)
    assert summary["frames"] == 4
    assert (tmp_path / "metrics.json").exists()


def test_boxes_are_finite():
    scene = SyntheticAerialScene(SceneConfig(frames=3))
    for frame in scene:
        assert np.isfinite(frame.boxes).all()
        assert np.isfinite(frame.velocities).all()
