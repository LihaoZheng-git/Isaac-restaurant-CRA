import json

import pytest

from robot.evaluation.restaurant_comparison import (
    comparison_plan, comparison_summary, contract_fingerprint, expected_stationary_rejection,
    extract_trial, trial_options,
)


def test_plan_pairs_all_methods_and_prespecifies_crossing_timing():
    pilot = comparison_plan()
    assert len(pilot) == 9
    study = comparison_plan(phase="study")
    assert len(study) == 75
    assert {t["crossing_delay_seconds"] for t in study} == {None, 7., 8., 9.}
    for case in {t["case"] for t in study}:
        for repeat in range(5):
            rows = [t for t in study if t["case"] == case and t["repeat"] == repeat]
            assert len({t["baseline"] for t in rows}) == 3
            assert len({contract_fingerprint(trial_options(t)) for t in rows}) == 1
    with pytest.raises(ValueError):
        comparison_plan(repetitions=0)


def test_runner_options_preserve_stationary_gate_and_human_scope():
    for trial in comparison_plan():
        options = trial_options(trial)
        assert "--motion" not in options  # runner adds it; component gate is still mandatory
        assert "--restaurant-route" in options
        assert "--human-carry-bowl" not in options
        if trial["case"] != "static_aisle":
            assert "--static-occupancy-memory-prior" not in options
            assert "--human-body-contact-prior" in options
        else:
            assert options[options.index("--stationary-frames")+1] == "60"
            assert "--sam-observed-context-box" in options


def test_missing_report_and_negative_clearance_are_retained(tmp_path):
    trial = comparison_plan()[0]
    missing = extract_trial(tmp_path, trial)
    assert not missing["report_available"] and not missing["passed"]
    (tmp_path / "stationary-gate.json").write_text(json.dumps({"passed": False}), encoding="utf-8")
    blocked = extract_trial(tmp_path, trial)
    assert blocked["reason"] == "stationary_gate_failed" and not blocked["motion_started"]
    (tmp_path / "report.json").write_text(json.dumps({"passed": False, "completed": False,
        "reason": "clearance_guard", "minimum_clearance_m": -.02}), encoding="utf-8")
    row = extract_trial(tmp_path, trial)
    assert row["monitor_overlap"]
    assert row["minimum_signed_monitor_clearance_m"] == -.02
    assert comparison_summary([row])["summary"][0]["attempts"] == 1
    assert not comparison_summary([row])["matched_cells"][0]["all_motion_reports"]
    with pytest.raises(ValueError, match="duplicate"):
        comparison_summary([row, row])


def test_matched_configuration_rejects_changed_camera_and_keeps_failures(tmp_path):
    trials = comparison_plan()[:3]
    rows = []
    for trial in trials:
        row = extract_trial(tmp_path, trial)
        row["report_available"] = True
        row["configuration"]["fixed_lambda"] = 1.25
        rows.append(row)
    summary = comparison_summary(rows)
    assert summary["matched_cells"][0]["matched_configuration"]
    assert all(s["passes"] == 0 and s["attempts"] == 1 for s in summary["summary"])
    rows[0]["configuration"]["experimental_camera_mount"] = {"upward_degrees": 20}
    assert comparison_summary(rows)["matched_cells"][0]["mismatches"] == ["experimental_camera_mount"]


def test_only_explicit_zero_wheel_gate_rejection_is_expected():
    message = "RuntimeError: stationary restaurant component gate failed; wheels are blocked"
    assert expected_stationary_rejection({"passed":False,"nonzero_wheel_commands":0}, message)
    assert not expected_stationary_rejection({"passed":False,"nonzero_wheel_commands":1}, message)
    assert not expected_stationary_rejection({"passed":False,"nonzero_wheel_commands":0}, "TypeError: logging failed")
    assert not expected_stationary_rejection({}, message)
