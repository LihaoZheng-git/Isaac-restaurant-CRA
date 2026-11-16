"""Bounded native mapping comparison keeps evidence and navigation inputs fixed."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluation.run_restaurant_demo import eligible_motion_case, prerequisite_matches


BASELINE = Path("configs/restaurant_boundary_response_baseline.json")
CANDIDATE = Path("configs/restaurant_boundary_response.json")
CASES = ("bin-static", "crate-static")
FLAG = "--restaurant-boundary-unit-flux"


def recipe(path):
    return json.loads(path.read_text())


def parse(monkeypatch, command, *, motion=False):
    script = Path(command[0]).resolve()
    argv = [*command, "--output", "outputs/unused-boundary-response-cli"]
    if motion:
        argv.append("--motion")
    monkeypatch.setattr("sys.argv", argv)
    namespace = {"__file__": str(script)}
    # Execute the actual CLI and guards; stop before filesystem/native effects.
    exec(compile(script.read_text().split("args.output.mkdir", 1)[0], str(script), "exec"), namespace)
    return namespace["args"]


@pytest.mark.parametrize("path", [BASELINE, CANDIDATE])
def test_recipe_retains_source_commands_and_declares_bounded_comparison(path):
    data = recipe(path)
    assert set(data["commands"]) == set(CASES)
    response = data["boundary_response"]
    assert not response["default_enabled"]
    assert response["ordinary_unknown_policy_unchanged"]
    assert response["native_causal_claim_requires_paired_baseline"]
    assert response["full_v3_controller_equivalence"] is False
    assert response["fixed_lambda"] == 1.25
    assert response["zero_flux_magnitude"] == .5
    for case, command in data["commands"].items():
        origin = data["derived_from"][case]
        original_path = Path(origin["path"])
        assert hashlib.sha256(original_path.read_bytes()).hexdigest() == origin["sha256"]
        original = json.loads(original_path.read_text())["commands"][case]
        assert hashlib.sha256(json.dumps(original, separators=(",", ":")).encode()).hexdigest() == origin["base_command_sha256"]
        assert command[:-4] == original
        assert command[-4:] == ["--restaurant-calibration", "kinetic", FLAG, str(response["unit_flux_magnitude"])]
        assert "--motion" not in command and "--output" not in command
        assert eligible_motion_case(case, command)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("motion", [False, True])
def test_real_cli_changes_only_boundary_endpoint_between_paired_cases(monkeypatch, case, motion):
    baseline = vars(parse(monkeypatch, recipe(BASELINE)["commands"][case], motion=motion))
    candidate = vars(parse(monkeypatch, recipe(CANDIDATE)["commands"][case], motion=motion))
    assert baseline["restaurant_boundary_unit_flux"] == 1.7
    assert candidate == {**baseline, "restaurant_boundary_unit_flux": 2.9}
    assert candidate["restaurant_calibration"] == "kinetic"
    assert candidate["mpc_horizon_steps"] == 12
    assert candidate["guidance_tracking_weight"] == 0
    assert candidate["stationary_frames"] == 60
    assert candidate["maximum_seconds"] == 45
    assert candidate["observe_crate_clearance"]
    assert candidate["crate_route_probe"]
    assert candidate["stationary_prop_world_pose"] == [15.65, 4.05, 0.]
    assert candidate["motion"] is motion


@pytest.mark.parametrize("case", CASES)
def test_candidate_cannot_reuse_baseline_stationary_gate(case):
    """Same source is insufficient when the parked boundary policy differs."""
    baseline = ["C:/isaacsim/python.bat", "-u", *recipe(BASELINE)["commands"][case]]
    candidate = ["C:/isaacsim/python.bat", "-u", *recipe(CANDIDATE)["commands"][case]]
    hashes = {"robot/source.py": "frozen"}
    scenes = {"scene.usda": "frozen-scene"}
    meta = dict(case=case, command=[*baseline, "--output", "parked-original"],
                source_sha256=hashes, scene_sha256=scenes)
    execution = dict(passed=True, source_unchanged=True)
    gate = dict(passed=True)
    assert prerequisite_matches(meta, execution, gate, case, "config", hashes, scenes,
                                expected_command=baseline)
    assert not prerequisite_matches(meta, execution, gate, case, "config", hashes, scenes,
                                    expected_command=candidate)


def test_boundary_experiment_rejects_launcher_case_aliases():
    command = recipe(CANDIDATE)["commands"]["crate-static"]
    assert not eligible_motion_case("crate-strong", command)
    assert not eligible_motion_case("bin-base", command)


def test_stronger_boundary_cannot_silently_enter_full_human_demo(monkeypatch, capsys):
    command = recipe(Path("configs/restaurant_service_risk_calibration_witnesses.json"))["commands"]["long-parallel"]
    with pytest.raises(SystemExit):
        parse(monkeypatch, [*command, FLAG, "2.9"])
    assert "isolated prop-route probes" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["nan", "inf", "1.6"])
def test_invalid_endpoint_is_rejected_before_native_start(monkeypatch, capsys, value):
    command = recipe(CANDIDATE)["commands"]["crate-static"].copy()
    command[command.index(FLAG) + 1] = value
    with pytest.raises(SystemExit):
        parse(monkeypatch, command)
    assert "finite and >=1.7" in capsys.readouterr().err
