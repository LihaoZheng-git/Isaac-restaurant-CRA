"""The support-proof variant must preserve the original scoring-only recipe."""

import hashlib
import json
from pathlib import Path

import pytest


BASE = Path("configs/restaurant_service_risk_calibration.json")
VARIANT = Path("configs/restaurant_service_risk_calibration_witnesses.json")
FLAG = "--unknown-distinct-surface-witnesses"


def test_witness_recipe_only_adds_explicit_proof_flag_to_existing_memory_modes():
    original = json.loads(BASE.read_text())
    variant = json.loads(VARIANT.read_text())
    assert variant["derived_from"]["sha256"] == hashlib.sha256(BASE.read_bytes()).hexdigest()
    assert variant["proof_variant"]["default_enabled"] is False
    assert variant["proof_variant"]["not_a_risk_only_trajectory_comparison"] is True
    assert variant["original_invocations"] == original["original_invocations"]
    assert variant["source_revision"] == original["source_revision"]
    assert set(variant["commands"]) == set(original["commands"])
    enabled = []
    for case, command in variant["commands"].items():
        assert FLAG not in original["commands"][case]
        if "--unknown-obstacle-planning" in command:
            assert command[-1] == FLAG and command.count(FLAG) == 1
            assert command[:-1] == original["commands"][case]
            enabled.append(case)
        else:
            assert command == original["commands"][case]
    assert set(enabled) == {"long-parallel", "long-parallel-near", "long-parallel-component"}


@pytest.mark.parametrize("case", ["long-parallel", "long-parallel-near", "long-parallel-component"])
@pytest.mark.parametrize("enabled", [False, True])
def test_witness_cli_is_explicit_and_parses_without_starting_isaac(monkeypatch, case, enabled):
    data = json.loads((VARIANT if enabled else BASE).read_text())
    command = data["commands"][case] + ["--output", "outputs/unused-witness-cli"]
    script = Path(command[0]).resolve()
    monkeypatch.setattr("sys.argv", command)
    namespace = {"__file__": str(script)}
    exec(compile(script.read_text().split("args.output.mkdir", 1)[0], str(script), "exec"), namespace)
    assert namespace["args"].unknown_distinct_surface_witnesses is enabled


@pytest.mark.parametrize("script", [
    "scripts/demos/cra_live_object_isaac_demo.py",
    "scripts/demos/restaurant_human_locomotion_isaac.py",
])
def test_witness_cli_rejects_enabling_proof_without_unknown_memory(monkeypatch, script):
    path = Path(script).resolve()
    monkeypatch.setattr("sys.argv", [script, FLAG, "--output", "outputs/unused-witness-cli"])
    with pytest.raises(SystemExit) as raised:
        exec(compile(path.read_text().split("args.output.mkdir", 1)[0], str(path), "exec"), {"__file__": str(path)})
    assert raised.value.code == 2
