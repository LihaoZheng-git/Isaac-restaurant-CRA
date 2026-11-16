"""The candidate recipe changes scoring, not the recorded V3 experiment."""

import hashlib
import json
from pathlib import Path

import pytest


RECIPE = Path("configs/restaurant_service_risk_calibration.json")


def recipe():
    return json.loads(RECIPE.read_text())


def original_command(command):
    command = command.copy()
    if command[0] == "scripts/demos/cra_live_object_isaac_demo.py":
        assert command[-2:] == ["--restaurant-calibration", "kinetic"]
        command = command[:-2]
    else:
        assert "--restaurant-calibration" not in command
    return command


def test_six_cases_match_all_seven_original_archived_invocation_command_hashes():
    data = recipe()
    assert len(data["commands"]) == 6 and len(data["original_invocations"]) == 7
    assert {row["stage"] for row in data["original_invocations"]} == {"component", "stationary", "motion"}
    for provenance in data["original_invocations"]:
        command = original_command(data["commands"][provenance["case"]])
        assert hashlib.sha256(json.dumps(command,separators=(",", ":")).encode()).hexdigest() == provenance["base_command_sha256"]
        assert "--motion" not in command and "--output" not in command
        assert "--long-service-human-gates" not in command


@pytest.mark.parametrize("case", ["long-parallel", "long-parallel-near", "bin-static", "crate-static"])
def test_candidate_perception_cli_parses_without_launching_isaac(monkeypatch, case):
    command = recipe()["commands"][case] + ["--output", "outputs/unused-risk-calibration-cli"]
    script = Path(command[0]).resolve()
    monkeypatch.setattr("sys.argv", command)
    namespace = {"__file__": str(script)}
    exec(compile(script.read_text().split("args.output.mkdir", 1)[0], str(script), "exec"), namespace)
    assert namespace["args"].restaurant_calibration == "kinetic"
    assert not namespace["args"].motion


def test_original_long_navigation_controller_and_unknown_policy_remain_explicit():
    command = recipe()["commands"]["long-parallel"]
    for flag, expected in (("--mpc-horizon-steps", "6"), ("--guidance-tracking-weight", "100.0"),
                           ("--object-factor-prior-risk", "0.25"), ("--simulation-rate", "0.025"),
                           ("--stationary-frames", "60"), ("--maximum-seconds", "300")):
        assert command[command.index(flag)+1] == expected
    assert "--observe-service-clearance" in command
    assert "--straight-service-walk" in command
    assert command[command.index("--restaurant-scene")+1].startswith("scenes/generated/service_pair_original_v3/")
