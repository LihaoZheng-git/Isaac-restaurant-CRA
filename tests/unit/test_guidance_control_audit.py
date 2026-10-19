"""Regression of the real PDE field and existing unicycle turn-seed trigger.

Authored geometry and phi are ORACLE. These tests never command a wheel.
"""

from dataclasses import replace

import numpy as np
import pytest

from robot.control.unicycle_model import lookahead_control_matrix
from robot.evaluation.guidance_control_audit import (
    GuidanceAuditConfig,
    build_audit_field,
    nominal_sequences,
    run_control_audit,
)
from robot.navigation.waypoint_controller import Waypoint, WaypointController
from robot.safety.safe_sage_cbf_filter import SafeSAGECBFSafetyFilter
from robot.safety.safe_sage_unicycle_mpc import (
    SafeSAGEUnicycleMPC,
    SafeSAGEUnicycleMPCConfig,
)


@pytest.fixture(scope="module")
def audit_fields():
    return {
        side: build_audit_field(side, GuidanceAuditConfig())
        for side in ("left", "right")
    }


@pytest.mark.parametrize("side,sign", [("left", 1), ("right", -1)])
def test_early_existing_turn_search_completes_with_unchanged_safety(
    audit_fields, side, sign
):
    result = run_control_audit(
        audit_fields[side],
        "unicycle",
        GuidanceAuditConfig(progress_seed_fraction=0.9, maximum_seconds=25),
    )
    assert result["passed"], result["reason"]
    assert all(result["checks"].values())
    xy = np.asarray(result["trajectory"])[:, :2]
    assert np.max(sign * xy[:, 1]) > 0.5
    assert np.linalg.norm(xy[-1] - [3.3, 0]) < 0.15
    assert result["fixed_lambda"] == 1.25
    assert result["nonzero_wheel_commands"] == 0
    assert "ORACLE" in result["scope"]
    assert any(
        p["status"] == "exact_safe_progress_seed_selected" for p in result["plans"]
    )
    for plan in result["plans"]:
        assert min(plan["dtcbf_margins"]) >= -1e-6
    # The pullback is B^T n, including yaw sensitivity. Its dual pairing must
    # equal the planar constraint for the *same* realized unicycle velocity.
    for row in result["records"]:
        matrix = lookahead_control_matrix(row["chassis"][2], lookahead_distance=0.1)
        u = np.asarray(row["executed"])
        assert np.dot(row["unicycle_cbf_normal"], u) == pytest.approx(
            np.dot(row["planar_cbf_normal"], matrix @ u), abs=1e-10
        )


def test_original_late_trigger_stall_is_not_reported_as_success(audit_fields):
    result = run_control_audit(
        audit_fields["left"], "unicycle", GuidanceAuditConfig(maximum_seconds=15)
    )
    assert not result["passed"]
    assert result["reason"] == "four_second_stationary_local_solution"
    assert not result["checks"]["arrived"]
    assert result["checks"]["executed_geometry"]
    assert result["checks"]["cbf_margins"]


def test_trigger_does_not_change_the_field_or_obstacle_unaware_nominal(audit_fields):
    early = build_audit_field(
        "left", replace(GuidanceAuditConfig(), progress_seed_fraction=0.9)
    )
    np.testing.assert_array_equal(
        early.h_full.values, audit_fields["left"].h_full.values
    )
    for a, b in zip(early.guidance_fields, audit_fields["left"].guidance_fields):
        np.testing.assert_array_equal(a.vectors_robot_xy, b.vectors_robot_xy)
    waypoint = WaypointController([Waypoint(3.3, 0)])
    wheel, planar = nominal_sequences(np.zeros(3), waypoint, 30)
    np.testing.assert_allclose(wheel, np.tile([0.2, 0], (30, 1)))
    np.testing.assert_allclose(planar, np.tile([0.2, 0, 0], (30, 1)))


def test_seed_feasibility_pruning_preserves_exact_winner(audit_fields, monkeypatch):
    field = audit_fields["left"]
    kwargs = dict(
        configuration_space=field.configuration_space,
        h_full=field.h_full,
        guidance_fields=field.guidance_fields,
        temporal_safety_derivative=0.0,
    )
    nominal = np.tile([0.2, 0.0], (30, 1))
    state = np.array([0.2, 0.0, 0.0])
    results = []
    for prune in (False, True):
        mpc = SafeSAGEUnicycleMPC(
            SafeSAGECBFSafetyFilter(),
            SafeSAGEUnicycleMPCConfig(
                horizon_steps=30,
                safety_substeps_per_interval=4,
                integration_substeps_per_interval=4,
                angular_velocity_weight=0.02,
                command_smoothness_weight=0.005,
                enable_recovery_feasibility_pruning=prune,
            ),
        )
        reference = mpc._evaluate_trajectory(state, np.zeros((30, 2)), **kwargs)
        calls = []
        original = mpc._inter_sample_unsafe_mask

        def counted(*args, **kw):
            calls.append(1)
            return original(*args, **kw)

        with monkeypatch.context() as patch:
            patch.setattr(mpc, "_inter_sample_unsafe_mask", counted)
            selected = mpc._select_progress_seed(
                reference,
                initial_state=state,
                nominal_sequence=nominal,
                target_chassis_states=None,
                **kwargs,
            )
        assert mpc._trajectory_is_safe(selected)
        assert mpc._objective_value(selected.commands, nominal) < mpc._objective_value(
            reference.commands, nominal
        )
        results.append((selected, len(calls)))
    np.testing.assert_array_equal(results[0][0].commands, results[1][0].commands)
    np.testing.assert_array_equal(results[0][0].margins, results[1][0].margins)
    assert results[1][1] < results[0][1]


@pytest.mark.parametrize(
    "changes",
    [
        {"progress_seed_fraction": 1},
        {"progress_seed_fraction": 0},
        {"progress_seed_fraction": float("nan")},
        {"execution_dt": 0.07},
        {"horizon_steps": True},
        {"solver_iterations": 1.5},
    ],
)
def test_invalid_audit_configuration_rejected(changes):
    with pytest.raises(ValueError):
        GuidanceAuditConfig(**changes)
