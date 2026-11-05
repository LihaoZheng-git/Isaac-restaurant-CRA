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
    assert {k: v for k, v in original.items() if k != "crate_delay_robust_bypass"} == {
        k: v for k, v in repaired.items() if k != "crate_delay_robust_bypass"
    }
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
