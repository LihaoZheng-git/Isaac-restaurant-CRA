from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_guidance_reference_speed_pool import fields, planner
from test_motion_passing_validation import example
from test_safe_sage_cbf_filter import make_fields

from robot.evaluation.crossing_strategy import summarize_crossing_strategy, summarize_wait_execution, zero_tangential_bias
from robot.evaluation.motion_passing_validation import summarize_motion_passing
from robot.safety.crossing_maneuver import apply_crossing_maneuver
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidanceParameters, SafeSAGEGuidancePolicy
from robot.safety.class_safety_policy import ClassSafetyParameters, LinearRiskFluxMapping


def setup(h=2., temporal=0., selected=True):
    m = planner(.25)
    kwargs = fields(h)
    result = m.plan(np.zeros(3), (.2, 0.), nominal_goal_grid_xy=(2., 0.), **kwargs)
    field = SimpleNamespace(pipeline=SimpleNamespace(**kwargs), temporal_safety_derivative=temporal,
        motion_passing_reference_track_ids=(7,) if selected else (),
        motion_passing_reference_sides=("left",) if selected else (),
        motion_passing_reference_source_timestamp_seconds=1. if selected else None,
        sensor_timestamp_seconds=2., sensor_frame_id=20)
    stop = SimpleNamespace(emergency_stop_active=False, safe_linear_velocity=0., safe_angular_velocity=0.)
    return m, result, field, stop


def test_yield_horizon_is_exact_checked_serializable_and_does_not_mutate_reference():
    m, original, field, stop = setup()
    before = original.command_sequence.copy()
    nominal = np.tile([.2, 0.], (m.config.horizon_steps, 1))
    result = apply_crossing_maneuver(m, original, np.zeros(3), nominal, field, "yield-then-go", lambda: stop)
    assert result.success and result.first_command == (0., 0.)
    assert not result.fallback_active  # live acceptance rejects emergency fallbacks
    assert result.minimum_dtcbf_margin >= -m.config.constraint_tolerance
    assert result.minimum_predicted_safety_value >= -m.config.constraint_tolerance
    assert result.crossing_maneuver["wait_exact_checked"]
    assert result.crossing_maneuver["wait_analytical_checked"]
    assert not result.crossing_maneuver["checked_reference_selected_for_plan"]
    assert not result.guidance_tracking.active
    assert result.crossing_maneuver["original_selection_timestamp_seconds"] == 1.
    np.testing.assert_array_equal(original.command_sequence, before)
    assert np.all(nominal[:, 0] == .2)
    assert not result.command_sequence.flags.writeable
    assert asdict(result)["crossing_maneuver"]["action"] == "checked_yield"


@pytest.mark.parametrize("failure", ["dynamic", "unsafe_geometry", "analytical_turn", "analytical_emergency"])
def test_wait_never_overrides_failed_dynamic_geometry_or_analytical_gate(failure):
    m, original, field, stop = setup()
    if failure == "dynamic":
        field.temporal_safety_derivative = -100.
    elif failure == "unsafe_geometry":
        cs, h, guides = make_fields(safety_value=2., unsafe_yaw_indices=(0, 1, 2, 3))
        field.pipeline = SimpleNamespace(configuration_space=cs, h_full=h, guidance_fields=guides)
    elif failure == "analytical_turn":
        stop.safe_angular_velocity = .1
    else:
        stop.emergency_stop_active = True
    result = apply_crossing_maneuver(m, original, np.zeros(3), (.2, 0.), field, "yield-then-go", lambda: stop)
    np.testing.assert_array_equal(result.command_sequence, original.command_sequence)
    assert result.crossing_maneuver["action"] == "original_safety_response_wait_unavailable"


def test_no_current_selection_cannot_create_wait_or_renew_original_epoch():
    m, original, field, _ = setup(selected=False)
    result = apply_crossing_maneuver(m, original, np.zeros(3), (.2, 0.), field,
        "rear-pass-or-yield", lambda: pytest.fail("stop check without a current selection"))
    np.testing.assert_array_equal(result.command_sequence, original.command_sequence)
    assert result.crossing_maneuver["original_selection_timestamp_seconds"] is None


def test_opposed_reference_uses_checked_wait_but_aligned_reference_is_preserved():
    m, original, field, stop = setup()
    detail = replace(original.guidance_tracking, active=True, reference_commands=((.1, .4),))
    commands = original.command_sequence.copy()
    commands[0, 1] = -.2
    opposed = replace(original, guidance_tracking=detail, command_sequence=commands)
    waited = apply_crossing_maneuver(m, opposed, np.zeros(3), (.2, 0.), field, "rear-pass-or-yield", lambda: stop)
    assert waited.crossing_maneuver["action"] == "checked_yield"
    commands[0, 1] = .2
    aligned = replace(opposed, command_sequence=commands)
    kept = apply_crossing_maneuver(m, aligned, np.zeros(3), (.2, 0.), field, "rear-pass-or-yield", lambda: stop)
    np.testing.assert_array_equal(kept.command_sequence, commands)
    assert kept.crossing_maneuver["checked_reference_selected_for_plan"]


def test_zero_bias_copies_all_class_and_track_policies_without_changing_normal_flux():
    p = SafeSAGEGuidanceParameters(ClassSafetyParameters(.6, 1.22), 1.25, "left")
    policy = SafeSAGEGuidancePolicy({"unknown": p, "person": p}, risk_flux_mapping=LinearRiskFluxMapping(.5, 1.7))
    audit = [dict(track_or_group_id=7, **{"lambda": 1.25, "boundary_flux": -1.22})]
    copied, applied, rows = zero_tangential_bias(policy, {7: p}, audit)
    assert all(v.tangential_strength == 0 for v in copied.class_parameters.values())
    assert applied[7].tangential_strength == 0 and p.tangential_strength == 1.25
    assert copied.parameters_for("unknown").boundary_flux == p.boundary_flux
    assert copied.parameters_for("person").safety_parameters == p.safety_parameters
    assert rows[0]["lambda"] == 0 and rows[0]["physical_CRA_lambda"] == 1.25
    assert audit[0]["lambda"] == 1.25


@pytest.mark.parametrize("mirror", [False, True])
def test_wait_go_uses_actual_crossing_not_closest_wait_and_keeps_old_failure(mirror):
    records, truth = example(False)
    # Wait below the crossing until the human is ahead; then cross and arrive.
    for row in truth:
        row["phase"] = "move"
        t = row["time"]
        row["robot_xy"][1] = min(t, 1.5)
    for t in np.linspace(4.05, 7., 60):
        truth.append(dict(time=float(t), phase="stopped_after", actor_xyz=[1., 2., 0.],
            robot_xy=[0., min(4., 1.5 + (t - 4.))]))
    if mirror:
        for row in truth:
            row["actor_xyz"][0] *= -1
        for row in records:
            row["motion_aware_passing"]["preferences"] = {"7": "right"}
            row["motion_aware_passing"]["selection_velocity_world_xy"] = [-.5, 0.]
            row["objects"][0]["diagnostics"]["centroid_world_xy"][0] *= -1
    legacy = summarize_motion_passing(records, truth, moving_robot=True, nominal_goal_world_xy=(0., 4.))
    assert not legacy["passed"]
    result = summarize_crossing_strategy(legacy, truth, strategy="yield-then-go", nominal_goal_world_xy=(0., 4.))
    assert result["passed"] and result["behind_projection_at_route_crossing_m"] == pytest.approx(-1.)
    fancy = summarize_crossing_strategy(legacy, truth, strategy="rear-pass-or-yield", nominal_goal_world_xy=(0., 4.))
    assert not fancy["passed"]
    truth[-1]["robot_xy"][1] = 3.
    failed = summarize_motion_passing(records, truth, moving_robot=True, nominal_goal_world_xy=(0., 4.))
    assert not summarize_crossing_strategy(failed, truth, strategy="yield-then-go", nominal_goal_world_xy=(0., 4.))["passed"]


@pytest.mark.parametrize("failure", [None, "rejected", "expired", "not_waiting", "not_checked"])
def test_wait_claim_requires_accepted_current_plans_and_observed_wait(failure):
    _, truth = example(False)
    for row in truth:
        row["phase"] = "move"
        row["robot_xy"][1] = min(row["time"], 1.) if failure != "not_waiting" else row["time"]
    plans = [dict(accepted=failure != "rejected", crossing_maneuver_ablation=dict(
        action="checked_yield", sensor_timestamp_seconds=t, sensor_frame_id=i,
        original_selection_timestamp_seconds=-10. if failure == "expired" else 1.,
        wait_exact_checked=failure != "not_checked", wait_analytical_checked=True))
        for i,t in enumerate([2., 2.2, 2.4])]
    result = summarize_wait_execution(plans, truth)
    assert result["passed"] is (failure is None)
