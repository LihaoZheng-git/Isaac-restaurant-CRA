from dataclasses import replace
import pickle

import numpy as np
import pytest
from test_crate_v3_controller import command, parse
from test_crate_progress_replay import saved_request
from test_safe_sage_cbf_filter import make_fields, make_filter
from test_safe_sage_unicycle_mpc import make_mpc

from robot.evaluation.crate_progress_diagnosis import nominal_for
from robot.safety.delay_robust_bypass import (
    DelayRobustBypassConfig,
    DelayRobustBypassMPC,
)
from robot.safety.laplace_guidance_tracking_mpc import LaplaceGuidanceTrackingConfig


@pytest.mark.parametrize("seconds", [-1, 0, 0.21, np.nan, np.inf, True])
def test_invalid_reserve_rejected(seconds):
    with pytest.raises(ValueError):
        DelayRobustBypassConfig(reserve_seconds=seconds)


def test_opt_in_does_not_change_base_recipe(monkeypatch):
    original = vars(
        parse(monkeypatch, command("restaurant_crate_v3_controller"))["args"]
    )
    repaired = vars(
        parse(monkeypatch, command("restaurant_crate_delay_robust_bypass"))["args"]
    )
    assert (
        not original["crate_delay_robust_bypass"]
        and repaired["crate_delay_robust_bypass"]
    )
    allowed = {
        "crate_delay_robust_bypass",
        "save_planner_replay",
        "save_failed_planner_replay",
    }
    assert {k: v for k, v in original.items() if k not in allowed} == {
        k: v for k, v in repaired.items() if k not in allowed
    }
    assert (
        repaired["save_planner_replay"] and not repaired["save_failed_planner_replay"]
    )
    values = command("restaurant_crate_delay_robust_bypass")
    values.remove("--crate-v3-controller")
    with pytest.raises(SystemExit):
        parse(monkeypatch, values)


def repaired():
    saved = saved_request()
    control = saved["control"]
    base = control.mpc
    control.mpc = DelayRobustBypassMPC(
        base.paper_filter, base.config, guidance_config=base.guidance_config
    )
    return saved


def test_online_plan_preserves_sources_nominal_and_original_execution_configuration():
    saved = repaired()
    control = saved["control"]
    state, _, field, _, _ = saved["request"]
    nominal, goal = nominal_for(state, 30, 0.2)
    original = nominal.copy()
    field_bytes = pickle.dumps(field)
    config = control.mpc.config
    result = control.plan(state, nominal, field, nominal_goal_grid_xy=goal)
    assert result.success and len(result.command_sequence) in (6, 30)
    assert result.delay_robust_bypass["reserve_seconds"] == 0.2
    np.testing.assert_array_equal(nominal, original)
    assert pickle.dumps(field) == field_bytes and control.mpc.config == config
    assert control.mpc.revalidate_commands(
        state,
        result.command_sequence,
        configuration_space=field.pipeline.configuration_space,
        h_full=field.pipeline.h_full,
        guidance_fields=field.pipeline.guidance_fields,
        temporal_safety_derivative=field.temporal_safety_derivative,
    )
    for steps in (6, 30):
        plain = control.mpc._planner(steps, reserve=False)
        reserved = control.mpc._planner(steps, reserve=True)
        assert plain.config == replace(config, horizon_steps=steps)
        assert (
            reserved.config.configuration_position_padding_m
            > plain.config.configuration_position_padding_m
        )
        assert (
            reserved.config.configuration_yaw_padding_radians
            > plain.config.configuration_yaw_padding_radians
        )


def test_full_long_tail_is_revalidated_not_truncated():
    mpc = make_mpc(horizon_steps=6, safety_substeps_per_interval=12)
    wrapper = DelayRobustBypassMPC(
        make_filter(),
        mpc.config,
        guidance_config=LaplaceGuidanceTrackingConfig(weight=10),
    )
    cs, h, g = make_fields(unsafe_yaw_indices=(2,), guidance=(1.0, 0.0))
    field = dict(
        configuration_space=cs,
        h_full=h,
        guidance_fields=g,
        temporal_safety_derivative=0.0,
    )
    commands = np.tile((0.1, 0.0), (30, 1))
    state = np.array([-1.0, 0.0, 0.0])
    assert wrapper.revalidate_commands(state, commands[:6], **field)
    assert not wrapper.revalidate_commands(state, commands, **field)
    assert not wrapper.revalidate_commands(state, commands[:7], **field)


def test_scope_rejects_human_prediction_and_short_nominal():
    saved = repaired()
    control = saved["control"]
    state, nominal, field, _, goal = saved["request"]
    with pytest.raises(ValueError, match="30-step nominal"):
        control.plan(state, nominal, field, nominal_goal_grid_xy=goal)
    control.mpc.predictive_geometry = object()
    with pytest.raises(ValueError, match="human"):
        control.mpc.plan(state, np.tile((0.2, 0.0), (30, 1)), nominal_goal_grid_xy=goal)


@pytest.mark.parametrize("blocked", ["none", "geometry", "analytical"])
def test_moving_pool_requires_full_geometry_and_unchanged_analytical_command(blocked):
    saved = repaired()
    wrapper = saved["control"].mpc
    planner = wrapper._planner(30, reserve=True, applied_command=(0.0, 0.0))
    cs, h, g = make_fields(
        safety_value=0.0 if blocked == "analytical" else 1.0,
        guidance=(-1.0, 0.0) if blocked == "analytical" else (1.0, 0.0),
        unsafe_yaw_indices=(0, 1, 2, 3) if blocked == "geometry" else (),
    )
    initial = (
        np.array([0.0, 0.0, 0.0])
        if blocked == "geometry"
        else np.array([-0.8, 1.0, 0.0])
    )
    nominal = np.tile((0.2, 0.0), (30, 1))
    goal = initial[:2] + (2.0, 0.0)
    field = dict(
        configuration_space=cs,
        h_full=h,
        guidance_fields=g,
        temporal_safety_derivative=0.0,
    )
    best, diagnostic = wrapper._moving_candidates(
        planner, initial, nominal, goal, field, saved["expected_result"]
    )
    assert diagnostic["tested"] == 75
    if blocked != "none":
        assert best is None and diagnostic["passed"] == 0
    else:
        assert best is not None and best[2].linear_velocity == pytest.approx(0.2)
        assert wrapper.revalidate_commands(initial, best[2].command_sequence, **field)
        assert best[2].status == "checked_static_moving_candidate_selected"
        assert not best[2].guidance_tracking.active
        assert best[2].guidance_tracking.reason == "separate_checked_static_arc_search"


def test_applied_command_reserve_keeps_physics_step_floor_and_original_limits():
    wrapper = repaired()["control"].mpc
    wrapper.config = replace(wrapper.config, integration_substeps_per_interval=12)
    np.testing.assert_allclose(
        wrapper._reserve((0.0, 0.0)), [0.25 * 0.2, 0.8 * 0.2 / 12]
    )
    np.testing.assert_array_equal(
        wrapper._reserve((0.2, 0.4)), wrapper._reserve((0.0, 0.0))
    )
    with pytest.raises(ValueError):
        wrapper._reserve((0.3, 0.0))


def test_applied_command_roundtrips_with_original_request(tmp_path):
    import json
    from robot.evaluation.planner_replay import save_planner_replay

    saved = repaired()
    control = saved["control"]
    state, _, field, _, _ = saved["request"]
    nominal, goal = nominal_for(state, 30, 0.2)
    options = dict(planning_applied_command=(0.1, 0.0))
    result = control.plan(state, nominal, field, nominal_goal_grid_xy=goal, **options)
    request = (state, nominal, field, None, goal, options)
    meta = save_planner_replay(tmp_path, control, request, result)
    assert meta["planning_applied_command"] == [0.1, 0.0]
    assert meta["prediction_state_timestamp_seconds"] is None
    assert json.loads((tmp_path / "planner-000000.json").read_text()) == meta
    restored = pickle.loads((tmp_path / meta["file"]).read_bytes())
    req = restored["request"]
    replay = restored["control"].plan(
        *req[:3], initial_command_guess=req[3], nominal_goal_grid_xy=req[4], **req[5]
    )
    np.testing.assert_allclose(replay.command_sequence, result.command_sequence)
    assert replay.delay_robust_bypass == result.delay_robust_bypass
    for options in (
        {"unexpected": 1},
        {"planning_applied_command": None},
        {"planning_applied_command": (0.1, np.nan)},
    ):
        with pytest.raises(ValueError):
            save_planner_replay(
                tmp_path / "invalid", control, (*request[:5], options), result
            )
    assert not (tmp_path / "invalid").exists()


def test_receipt_probe_rejects_horizon_that_only_passes_at_request():
    wrapper = repaired()["control"].mpc
    cs, h, g = make_fields(unsafe_yaw_indices=(2,), guidance=(1.0, 0.0))
    field = dict(
        configuration_space=cs,
        h_full=h,
        guidance_fields=g,
        temporal_safety_derivative=0.0,
    )
    commands = np.tile((0.0, 0.0), (30, 1))
    safe = np.array([-1.0, 0.0, 0.0])
    unsafe = np.array([0.0, 0.0, 0.0])
    assert wrapper.revalidate_commands(safe, commands, **field)
    assert not wrapper._delay_checked((safe, unsafe), commands, field)
    poses = wrapper._delay_poses(safe, (0.2, 0.4))
    assert len(poses) == 6 and poses[1][0] > safe[0] and poses[1][2] > safe[2]
    np.testing.assert_array_equal(poses[0], safe)


def test_checked_moving_continuation_avoids_redundant_cold_recovery(monkeypatch):
    from robot.safety.laplace_guidance_tracking_mpc import LaplaceGuidanceTrackingMPC

    saved = repaired()
    c = saved["control"]
    state, _, field, _, _ = saved["request"]
    nominal, goal = nominal_for(state, 30, 0.2)
    calls = []
    original = LaplaceGuidanceTrackingMPC.plan

    def record(self, *args, **kwargs):
        calls.append(
            (self.config.horizon_steps, kwargs.get("initial_command_guess") is not None)
        )
        result = original(self, *args, **kwargs)
        # Exercise the long continuation branch even on this free-space field.
        return replace(
            result, guidance_tracking=replace(result.guidance_tracking, active=False)
        )

    monkeypatch.setattr(LaplaceGuidanceTrackingMPC, "plan", record)
    result = c.plan(
        state,
        nominal,
        field,
        nominal_goal_grid_xy=goal,
        initial_command_guess=nominal.copy(),
        planning_applied_command=(0.1, 0.0),
    )
    assert result.success and result.delay_robust_bypass["checked_moving_continuation"]
    assert calls == [(6, True), (30, True)]
    assert result.delay_robust_bypass["receipt_pose_probe_count"] == 6
