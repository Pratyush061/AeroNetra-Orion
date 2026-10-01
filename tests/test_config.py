import pytest

from orion.config import OrionConfig, config_from_dict, load_config, to_dict


def test_defaults():
    cfg = load_config(None)
    assert isinstance(cfg, OrionConfig)
    assert cfg.events.threshold > 0
    assert cfg.scene.frames > 0


def test_yaml_override(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "events:\n  threshold: 0.5\n"
        "detect:\n  min_area: 99\n"
        "output_dir: out\n"
    )
    cfg = load_config(path)
    assert cfg.events.threshold == 0.5
    assert cfg.detect.min_area == 99
    assert cfg.output_dir == "out"
    # a nested default that was not overridden must survive
    assert cfg.track.max_age == 10


def test_dict_roundtrip():
    cfg = config_from_dict({"scene": {"frames": 5}})
    d = to_dict(cfg)
    assert d["scene"]["frames"] == 5
    assert d["events"]["threshold"] == OrionConfig().events.threshold


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/definitely/not/here.yaml")


def test_non_mapping_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError):
        load_config(path)
