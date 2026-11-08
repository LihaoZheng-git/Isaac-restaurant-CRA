"""A new placement cannot alter control or reuse the previous parked proof."""

import hashlib
import json
from pathlib import Path

import pytest

from test_restaurant_boundary_response_recipe import parse
from scripts.evaluation.run_restaurant_demo import eligible_motion_case, prerequisite_matches


BASELINE = Path("configs/restaurant_boundary_response_baseline.json")
CLOSER = Path("configs/restaurant_crate_closer_calibrated.json")


def commands():
    original = json.loads(BASELINE.read_text())["commands"]["crate-static"]
    closer = json.loads(CLOSER.read_text())["commands"]["crate-static"]
    return original, closer


@pytest.mark.parametrize("motion", [False, True])
def test_closer_placement_keeps_the_real_controller_and_score_configuration(monkeypatch, motion):
    original, closer = commands()
    baseline = vars(parse(monkeypatch, original, motion=motion))
    candidate = vars(parse(monkeypatch, closer, motion=motion))
    assert candidate == {
        **baseline,
        "restaurant_scene": baseline["restaurant_scene"].with_name("restaurant_crate_route_setback3.usda"),
    }
    assert candidate["restaurant_boundary_unit_flux"] == 1.7
    assert candidate["restaurant_calibration"] == "kinetic"
    assert candidate["mpc_horizon_steps"] == 12
    assert candidate["guidance_tracking_weight"] == 0
    assert candidate["stationary_frames"] == 60
    assert candidate["observe_crate_clearance"]
    assert eligible_motion_case("crate-static", closer)
    closer.remove("--observe-crate-clearance")
    with pytest.raises(SystemExit):
        parse(monkeypatch, closer, motion=motion)


def test_previous_placement_cannot_supply_the_new_stationary_prerequisite():
    original, closer = commands()
    baseline = ["C:/isaacsim/python.bat", "-u", *original]
    candidate = ["C:/isaacsim/python.bat", "-u", *closer]
    hashes = {"robot/source.py": "unchanged"}
    scenes = {"scene.usda": "same-for-command-isolation"}
    meta = dict(case="crate-static", command=[*baseline, "--output", "parked-original"],
                source_sha256=hashes, scene_sha256=scenes)
    execution = dict(passed=True, source_unchanged=True)
    gate = dict(passed=True)
    assert prerequisite_matches(meta, execution, gate, "crate-static", "config", hashes, scenes,
                                expected_command=baseline)
    assert not prerequisite_matches(meta, execution, gate, "crate-static", "config", hashes, scenes,
                                    expected_command=candidate)


def test_recipe_records_unchanged_base_and_bounded_placement():
    data = json.loads(CLOSER.read_text())
    assert set(data["commands"]) == {"crate-static"}
    assert data["derived_from"]["sha256"] == hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    assert data["placement"]["crate_center_world_xy"] == [17.5, 3.42]
    assert data["placement"]["closer_than_baseline_metres"] == .02
    assert data["placement"]["driven_trial_limit"] == 1
