import numpy as np
import pytest

from orion.datasets import (
    get_dataset,
    is_available,
    list_datasets,
    load_events_npz,
    parse_auair,
    parse_uavdt_gt,
    parse_visdrone_det,
    require,
)
from orion.events import EventStream


def test_registry_has_event_and_aerial():
    keys = {d.key for d in list_datasets()}
    assert {"dsec", "mvsec", "visevent", "auair", "uavdt", "seadronessee", "visdrone"} <= keys
    assert any(d.kind == "event" for d in list_datasets())
    assert any(d.kind == "aerial" for d in list_datasets())


def test_get_dataset_unknown():
    with pytest.raises(KeyError):
        get_dataset("does-not-exist")


def test_parse_visdrone_skips_ignored_regions():
    lines = ["10,20,30,40,1,1,0,0", "5,5,10,10,1,0,0,0"]
    assert parse_visdrone_det(lines) == [(10.0, 20.0, 40.0, 60.0)]


def test_parse_uavdt():
    lines = ["1,1,10,20,30,40,0,0,0"]
    assert parse_uavdt_gt(lines) == [(10.0, 20.0, 40.0, 60.0)]


def test_parse_auair():
    obj = {"annotations": [{"bbox": {"left": 1, "top": 2, "width": 3, "height": 4}}]}
    assert parse_auair(obj) == [(1.0, 2.0, 4.0, 6.0)]


def test_load_events_npz(tmp_path):
    path = tmp_path / "events.npz"
    np.savez(
        path,
        x=np.array([1, 2], np.int32),
        y=np.array([3, 4], np.int32),
        t=np.array([0.0, 0.1]),
        p=np.array([1, -1], np.int8),
        width=10,
        height=8,
    )
    ev = load_events_npz(path)
    assert isinstance(ev, EventStream)
    assert len(ev) == 2
    assert ev.width == 10 and ev.height == 8


def test_load_events_npz_missing_keys(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, x=np.array([1]), y=np.array([1]))
    with pytest.raises(ValueError):
        load_events_npz(path)


def test_require_missing_dataset(tmp_path):
    assert is_available("dsec", tmp_path) is False
    with pytest.raises(FileNotFoundError):
        require("dsec", tmp_path)
