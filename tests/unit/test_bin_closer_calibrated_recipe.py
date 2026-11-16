"""The closer bin changes placement, not scoring, control or parked-proof identity."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluation.run_restaurant_demo import eligible_motion_case, prerequisite_matches
from test_restaurant_boundary_response_recipe import parse

BASELINE = Path("configs/restaurant_boundary_response_baseline.json")
CLOSER = Path("configs/restaurant_bin_closer_calibrated.json")


def commands():
    return (json.loads(path.read_text())["commands"]["bin-static"]
            for path in (BASELINE, CLOSER))


@pytest.mark.parametrize("motion", [False, True])
def test_closer_bin_changes_only_real_cli_scene(monkeypatch, motion):
    original, candidate = commands()
    before = vars(parse(monkeypatch, original, motion=motion))
    after = vars(parse(monkeypatch, candidate, motion=motion))
    assert after == {**before, "restaurant_scene": before["restaurant_scene"].with_name(
        "restaurant_bin_closer25_pass.usda")}
    assert after["restaurant_boundary_unit_flux"] == 1.7
    assert after["restaurant_calibration"] == "kinetic"
    assert after["mpc_horizon_steps"] == 12 and after["guidance_tracking_weight"] == 0
    assert after["stationary_prop_target"] == "ServiceContainer"
    assert after["stationary_frames"] == 60 and after["observe_crate_clearance"]
    assert eligible_motion_case("bin-static", candidate)


@pytest.mark.parametrize("missing", ["--confirmed-prop-refresh", "--static-prop-evidence",
                                    "--observe-prop-clearance"])
def test_closer_bin_keeps_existing_prerequisite_guards(monkeypatch, missing):
    _, candidate = commands()
    candidate.remove(missing)
    with pytest.raises(SystemExit):
        parse(monkeypatch, candidate, motion=True)


@pytest.mark.parametrize("change", ["target", "scene"])
def test_closer_bin_cannot_admit_wrong_target_or_arbitrary_scene(monkeypatch, change):
    _, candidate = commands()
    if change == "target":
        candidate[candidate.index("--stationary-prop-target") + 1] = "BrokenCrate"
    else:
        candidate[candidate.index("--restaurant-scene") + 1] = "unverified_bin_scene.usda"
    with pytest.raises(SystemExit):
        parse(monkeypatch, candidate, motion=True)


def test_previous_bin_placement_cannot_supply_stationary_prerequisite():
    original, candidate = commands()
    before = ["C:/isaacsim/python.bat", "-u", *original]
    after = ["C:/isaacsim/python.bat", "-u", *candidate]
    hashes, scenes = {"robot/source.py": "unchanged"}, {"scene.usda": "same-for-command-isolation"}
    meta = dict(case="bin-static", command=[*before, "--output", "parked-original"],
                source_sha256=hashes, scene_sha256=scenes)
    execution, gate = dict(passed=True, source_unchanged=True), dict(passed=True)
    assert prerequisite_matches(meta, execution, gate, "bin-static", "config", hashes, scenes,
                                expected_command=before)
    assert not prerequisite_matches(meta, execution, gate, "bin-static", "config", hashes, scenes,
                                    expected_command=after)


def test_recipe_has_one_bounded_placement_and_unchanged_base():
    data = json.loads(CLOSER.read_text())
    assert set(data["commands"]) == {"bin-static"}
    assert data["derived_from"]["sha256"] == hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    assert data["placement"]["bin_center_world_xy"] == [17.5, 3.65]
    assert data["placement"]["closer_than_baseline_metres"] == .25
    assert data["placement"]["driven_trial_limit"] == 1
