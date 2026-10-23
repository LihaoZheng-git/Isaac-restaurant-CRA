from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_laplace_guidance_tracking_mpc import config, field
from test_safe_sage_cbf_filter import make_filter

from robot.control.unicycle_model import lookahead_state, lookahead_control_matrix
from robot.safety.checked_rear_pass_reference import (
    CheckedRearPassReferenceConfig,
    rear_pass_context,
    search_checked_rear_pass_reference,
)
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPC
from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter


@pytest.mark.parametrize(
    "changes",
    [
        dict(offset_m=0),
        dict(offset_m=np.nan),
        dict(lookahead_m=True),
        dict(candidate_limit=577),
        dict(candidate_limit=1.5),
        dict(candidate_limit=True),
        dict(goal_fade_distance_m=0.05),
        dict(existing_turn_heading_radians=0),
        dict(existing_turn_heading_radians=np.pi / 2),
    ],
)
def test_invalid_reference_configuration_rejected(changes):
    with pytest.raises(ValueError):
        CheckedRearPassReferenceConfig(**changes)


@pytest.mark.parametrize("side", [-1.0, 1.0])
@pytest.mark.parametrize("yaw", [0.0, 1.1])
def test_trailing_target_mirrors_current_motion_and_rotates_with_route(side, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    result = rear_pass_context(
        [0.0, 0.0, yaw],
        rot @ [3.9, 0.0],
        rot @ [3.3, -side * 0.5],
        rot @ [0.0, side * 0.5],
        rot @ [1.0, 0.0],
    )
    assert result["mode"] == "rear"
    np.testing.assert_allclose(
        result["target_grid_xy"], rot @ [1.0, -side * 0.7], atol=1e-12
    )


def test_lead_wait_existing_detour_and_outgoing_release_are_behavior_gates():
    args = ([0.0, 0.0, 0.0], [3.9, 0.0], [1.0, -0.5], [0.0, 0.5], [1.0, 0.0])
    assert rear_pass_context(*args)["mode"] == "wait"
    assert rear_pass_context([0.0, -0.2, 0.0], *args[1:])["mode"] == "rear"
    result = rear_pass_context(
        [0.0, 0.0, 0.0], [3.9, 0.0], [3.3, 1.3], [0.0, 0.5], [1.0, 0.0]
    )
    assert result["mode"] == "none" and "cleared" in result["reason"]


def test_target_fades_toward_original_waypoint_and_never_claims_it_reached():
    result = rear_pass_context(
        [0.0, 0.0, 0.0], [0.2, 0.0], [3.3, -0.5], [0.0, 0.5], [1.0, 0.0]
    )
    np.testing.assert_allclose(result["target_grid_xy"], [0.2, -0.7 * (0.2 / 0.8) ** 2])
    near = rear_pass_context(
        [0.15, 0.0, 0.0], [0.2, 0.0], [3.3, -0.5], [0.0, 0.5], [1.0, 0.0]
    )
    assert near["mode"] == "none" and near["reason"] == "nominal_waypoint_region"


@pytest.mark.parametrize("side", [-1, 1])
def test_opt_in_current_trailing_turn_can_continue_before_lateral_translation(side):
    args = ([0, 0, -side * 0.35], [3.9, 0], [1, -side * 0.5], [0, side * 0.5], [1, 0])
    assert rear_pass_context(*args)["mode"] == "wait"
    result = rear_pass_context(
        *args, CheckedRearPassReferenceConfig(existing_turn_heading_radians=0.2)
    )
    assert result["mode"] == "rear" and result["continuing_existing_trailing_turn"]
    assert result["reason"] == "continue_existing_trailing_turn"
    assert (
        rear_pass_context(
            [0, 0, side * 0.35],
            *args[1:],
            CheckedRearPassReferenceConfig(existing_turn_heading_radians=0.2),
        )["mode"]
        == "wait"
    )  # Turning toward the pedestrian is not trailing commitment.


@pytest.mark.parametrize("velocity", [(0.0, 0.0), (0.5, 0.0), (0.0, 3.0)])
def test_unsupported_motion_cannot_create_a_trailing_reference(velocity):
    assert (
        rear_pass_context(
            [0.0, 0.0, 0.0], [3.9, 0.0], [3.3, -0.5], velocity, [1.0, 0.0]
        )["mode"]
        == "none"
    )


def search(*, failure=None, budget=576):
    mpc = SafeSAGEUnicycleMPC(make_filter(), replace(config(), horizon_steps=6))
    options = {
        **field(
            g=(-1.0, 0.0) if failure == "future" else (1.0, 0.0),
            h=0.001 if failure == "future" else 2.0,
        ),
        "temporal_safety_derivative": -100.0 if failure == "temporal" else 0.0,
    }
    backup = SafeSAGEUnicycleCBFFilter(mpc.paper_filter)
    initial = np.zeros(3)

    def check(command):
        if failure in {"analytical", "modified"}:
            return SimpleNamespace(
                emergency_stop_active=failure == "analytical",
                safe_linear_velocity=0.0,
                safe_angular_velocity=0.0,
            )
        q = lookahead_state(initial, lookahead_distance=mpc.config.lookahead_distance)
        return backup.filter_command(
            *command, query_position_grid_xy=q[:2], query_yaw_radians=q[2], **options
        )

    result = search_checked_rear_pass_reference(
        mpc,
        initial,
        [2.0, 0.0],
        [3.3, -0.5],
        [0.0, 0.5],
        [1.0, 0.0],
        options,
        check,
        config=CheckedRearPassReferenceConfig(candidate_limit=budget),
    )
    return mpc, result, options, check


def test_complete_productive_reference_passes_exact_and_analytical_checks_without_field_mutation():
    mpc, result, options, check = search()
    value = result.trajectory
    assert value is not None and result.diagnostics["selected"]
    assert result.diagnostics["reason"] == "checked_reference_selected"
    assert len(value.commands) == mpc.config.horizon_steps
    assert np.linalg.norm(value.chassis_states[-1, :2]) >= 0.01
    verified = mpc._evaluate_trajectory(
        np.zeros(3), value.commands, require_feasible=True, **options
    )
    assert verified is not None and mpc._trajectory_is_safe(verified)
    first = check(value.commands[0])
    assert not first.emergency_stop_active
    np.testing.assert_allclose(
        value.commands[0],
        [first.safe_linear_velocity, first.safe_angular_velocity],
        atol=1e-6,
    )
    assert (
        np.all(options["h_full"].values == 2.0)
        and options["temporal_safety_derivative"] == 0.0
    )


@pytest.mark.parametrize("failure", ["temporal", "analytical", "modified"])
def test_failed_temporal_or_first_analytical_acceptance_never_returns_a_reference(
    failure,
):
    _, result, _, _ = search(failure=failure)
    assert result.trajectory is None and not result.diagnostics["selected"]
    assert result.diagnostics["reason"] == "no_complete_checked_reference"


def test_budget_limit_can_return_only_a_complete_checked_horizon():
    mpc, result, options, _ = search(budget=1)
    assert result.diagnostics["counts"]["tested"] == 1
    if result.trajectory is not None:
        assert len(result.trajectory.commands) == mpc.config.horizon_steps
        assert mpc._trajectory_is_safe(
            mpc._evaluate_trajectory(np.zeros(3), result.trajectory.commands, **options)
        )


def test_wait_gate_does_not_invoke_search_or_invent_a_safe_command():
    result = search_checked_rear_pass_reference(
        None,
        [0.0, 0.0, 0.0],
        [2.0, 0.0],
        [1.0, -0.5],
        [0.0, 0.5],
        [1.0, 0.0],
        {},
        lambda _: pytest.fail("ineligible maneuver cannot filter a command"),
    )
    assert result.trajectory is None and result.diagnostics["context"]["mode"] == "wait"
    assert result.diagnostics["reason"] == "insufficient_maneuver_lead"
    assert result.diagnostics["counts"]["tested"] == 0


def test_future_semantic_constraints_are_checked_without_replacing_exact_dynamics():
    mpc, result, _, _ = search(failure="future")
    assert result.diagnostics["counts"]["exact_feasible"] > 0
    assert result.diagnostics["counts"]["future_analytical_rejections"] > 0
    if result.trajectory is not None:
        value = result.trajectory
        for j in range(mpc.config.horizon_steps):
            normal = (
                lookahead_control_matrix(
                    value.safety_states[j, 2],
                    lookahead_distance=mpc.config.lookahead_distance,
                ).T
                @ value.samples.guidance_normals[j]
            )
            rhs = (
                -mpc.paper_filter.config.gamma * max(0.0, value.safety_values[j])
                - value.samples.scaled_temporal_derivatives[j]
            )
            assert (
                normal @ value.commands[j]
                >= rhs - mpc.paper_filter.config.numerical_tolerance
            )
