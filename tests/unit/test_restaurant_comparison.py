import json

import pytest

from robot.evaluation.restaurant_comparison import (
    comparison_plan, comparison_summary, contract_fingerprint, expected_stationary_rejection,
    extract_trial, trial_options, planner_result_diagnostic,
)


def test_failed_plan_margin_remains_unavailable_in_strict_json():
    import numpy as np
    from test_safe_sage_unicycle_mpc import make_mpc

    mpc = make_mpc()
    result = mpc._fallback_result(np.zeros(3), np.tile([.2, 0.], (10, 1)),
                                  status="declared rejected horizon")
    record = {"accepted": False, "reason": result.status,
              **planner_result_diagnostic(result)}
    decoded = json.loads(json.dumps(record, allow_nan=False))
    assert decoded["minimum_dtcbf_margin"] is None
    assert decoded["minimum_dtcbf_margin_status"] == "unavailable_nonfinite"
    assert not decoded["accepted"] and not decoded["solver_success"]
    assert decoded["solver_fallback_active"]
    assert decoded["reason"] == "declared rejected horizon"
    assert np.isnan(result.minimum_dtcbf_margin)  # reporting never mutates control


def test_absent_and_finite_plan_diagnostics_are_distinct():
    from test_safe_sage_unicycle_mpc import plan_synthetic

    absent = planner_result_diagnostic(None)
    assert absent["minimum_dtcbf_margin_status"] == "unavailable_no_result"
    result = plan_synthetic(guidance=(0, 0), gradient=(0, 0, 0), safety_value=2.)
    record = planner_result_diagnostic(result)
    assert record["minimum_dtcbf_margin"] == result.minimum_dtcbf_margin
    assert record["minimum_dtcbf_margin_status"] == "finite"
    json.dumps([absent, record], allow_nan=False)


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
            assert options[options.index("--human-cluster-radius")+1] == ".18"
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


def test_arrival_does_not_hide_failed_perception_acceptance(tmp_path):
    (tmp_path/"report.json").write_text(json.dumps({"completed":True,"passed":False,
        "reason":"goal_reached","human_navigation_validation":{"checks":{"kinetic_coverage":False}}}))
    row=extract_trial(tmp_path,comparison_plan()[0])
    assert row["arrived"] and not row["passed"]
    assert row["reason"] == "acceptance_failed_after_arrival"
    assert row["failed_validation_checks"] == {"human_navigation_validation":["kinetic_coverage"]}
