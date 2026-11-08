"""The new bin layer preserves source assets and the original matched-bin output."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import test_service_pair_layout as service_fixture

source_root = service_fixture.source_root

BUILDER_PATH = Path(__file__).resolve().parents[2] / "scripts/scene_creation/build_crate_route_probe.py"
spec = importlib.util.spec_from_file_location("closer_bin_builder", BUILDER_PATH)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

from pxr import Usd  # noqa: E402


def hashes(root):
    return {p: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_both_bin_layers_preserve_sources_size_height_and_old_output(source_root, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", source_root)
    generated = source_root / "scenes/generated"
    base = json.loads((generated / "restaurant_service_demo_v3_open_aisle_bench.json").read_text())
    original = next(p for p in base["props"] if p["name"] == "ServiceContainer")
    original_size = np.array(original["bounds_maximum"]) - original["bounds_minimum"]
    for closer, name, y in ((False, "restaurant_bin_matched_pass", 3.4),
                            (True, "restaurant_bin_closer25_pass", 3.65)):
        before = hashes(source_root)
        builder.build(setback_cm=5, matched_bin=True, closer_bin=closer)
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == sha for p, sha in before.items())
        result = json.loads((generated / (name + ".json")).read_text())
        assert result["bin_center_world_xy"] == pytest.approx([17.5, y])
        prop = next(p for p in result["props"] if p["name"] == "ServiceContainer")
        np.testing.assert_allclose(np.array(prop["bounds_maximum"]) - prop["bounds_minimum"],
                                   original_size, rtol=0, atol=1e-9)
        assert prop["bounds_minimum"][2] == original["bounds_minimum"][2]
        assert prop["bounds_maximum"][2] == original["bounds_maximum"][2]
        stage = Usd.Stage.Open(result["scene"])
        assert not stage.GetPrimAtPath(builder.DEMO_ROOT + "/BrokenCrate").IsActive()
        saved = hashes(source_root)
        with pytest.raises(ValueError, match="new output paths"):
            builder.build(setback_cm=5, matched_bin=True, closer_bin=closer)
        assert hashes(source_root) == saved


@pytest.mark.parametrize("matched,setback", [(False, 0), (False, 5), (True, 0), (True, 3)])
def test_closer_bin_rejects_unrelated_modes_before_writing(source_root, monkeypatch, matched, setback):
    monkeypatch.setattr(builder, "ROOT", source_root)
    before = hashes(source_root)
    with pytest.raises(ValueError, match="closer bin requires"):
        builder.build(setback_cm=setback, matched_bin=matched, closer_bin=True)
    assert hashes(source_root) == before
