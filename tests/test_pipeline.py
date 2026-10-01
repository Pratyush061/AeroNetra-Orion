import numpy as np

from orion.config import OrionConfig
from orion.pipeline import OrionPipeline, run_demo
from orion.scene import SyntheticAerialScene


def _small_cfg():
    cfg = OrionConfig()
    cfg.scene.width = 160
    cfg.scene.height = 120
    cfg.scene.frames = 8
    cfg.scene.num_vehicles = 3
    cfg.scene.num_intruders = 1
    cfg.save_video = False
    return cfg


def test_scene_is_deterministic():
    cfg = _small_cfg()
    a = list(SyntheticAerialScene(cfg.scene))
    b = list(SyntheticAerialScene(cfg.scene))
    assert np.array_equal(a[0].gray, b[0].gray)
    assert np.array_equal(a[-1].boxes, b[-1].boxes)


def test_pipeline_runs_and_writes_metrics(tmp_path):
    cfg = _small_cfg()
    summary = run_demo(cfg, out_dir=tmp_path)
    assert summary["frames"] == 8
    for key in ("critical_alerts", "warning_alerts", "min_ttc_s", "mean_recall"):
        assert key in summary
    assert (tmp_path / "metrics.json").exists()


def test_pipeline_reset_clears_state(tmp_path):
    cfg = _small_cfg()
    pipeline = OrionPipeline(cfg)
    pipeline.run(SyntheticAerialScene(cfg.scene), out_dir=tmp_path)
    pipeline.reset()
    assert pipeline.tracker.tracks == []
