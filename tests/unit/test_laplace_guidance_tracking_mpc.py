from dataclasses import fields, replace

import numpy as np
import pytest
from test_safe_sage_cbf_filter import make_fields, make_filter

from robot.safety.laplace_guidance_tracking_mpc import (
    LaplaceGuidanceTrackingConfig,
    LaplaceGuidanceTrackingMPC,
    _ReferenceTrackingMPC,
)
from robot.safety.safe_sage_unicycle_mpc import (
    SafeSAGEUnicycleMPC,
    SafeSAGEUnicycleMPCConfig,
)


def config():
    return SafeSAGEUnicycleMPCConfig(
        horizon_steps=8, maximum_solver_iterations=1000, angular_velocity_weight=0.02
    )


def field(g=(1.0, 1.0), h=0.3):
    cs, values, guidance = make_fields(guidance=g, safety_value=h)
    return dict(configuration_space=cs, h_full=values, guidance_fields=guidance)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, True, "1"])
def test_invalid_weight_is_rejected(value):
    with pytest.raises(ValueError):
        LaplaceGuidanceTrackingConfig(weight=value)


@pytest.mark.parametrize("value", [0, -1, np.nan, True])
def test_invalid_activation_is_rejected(value):
    with pytest.raises(ValueError):
        LaplaceGuidanceTrackingConfig(activation_safety_value=value)


def test_zero_weight_is_identical_to_primary():
    c, paper = config(), make_filter()
    a = SafeSAGEUnicycleMPC(paper, c).plan(np.zeros(3), (0.2, 0), **field())
    b = LaplaceGuidanceTrackingMPC(paper, c).plan(np.zeros(3), (0.2, 0), **field())
    assert type(a) is type(b)
    for f in fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y)
        else:
            assert x == y


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_guidance_reference_and_execution_are_checked_and_nominal_unchanged(side):
    c, paper = config(), make_filter()
    m = LaplaceGuidanceTrackingMPC(
        paper, c, guidance_config=LaplaceGuidanceTrackingConfig(weight=10)
    )
    nominal = np.tile([0.2, 0.0], (c.horizon_steps, 1))
    before = nominal.copy()
    kwargs = field(g=(1.0, side))
    r = m.plan(np.zeros(3), nominal, nominal_goal_grid_xy=(2.0, 0.0), **kwargs)
    np.testing.assert_array_equal(nominal, before)
    d = r.guidance_tracking
    assert r.success and d.active and d.nominal_unchanged
    assert side * r.angular_velocity > 0.01
    assert d.reference_goal_progress_m > 0
    ref = m._evaluate_trajectory(
        np.zeros(3),
        np.array(d.reference_commands),
        temporal_safety_derivative=0.0,
        **kwargs,
    )
    assert ref is not None and m._trajectory_is_safe(ref)
    assert r.minimum_dtcbf_margin >= -c.constraint_tolerance
    assert d.effective_position_weight > 0 and d.effective_yaw_weight > 0


def test_frame_rotation_preserves_control():
    outputs = []
    for yaw in (0.0, np.pi / 2, -np.pi + 0.1):
        goal = 2 * np.array([np.cos(yaw), np.sin(yaw)])
        direction = yaw + np.pi / 4
        m = LaplaceGuidanceTrackingMPC(
            make_filter(),
            config(),
            guidance_config=LaplaceGuidanceTrackingConfig(weight=10),
        )
        r = m.plan(
            np.array([0.0, 0.0, yaw]),
            (0.2, 0.0),
            nominal_goal_grid_xy=goal,
            **field(g=(np.cos(direction), np.sin(direction))),
        )
        assert r.success
        outputs.append(r.first_command)
    np.testing.assert_allclose(outputs, [outputs[0]] * 3, atol=1e-5)


def test_no_reference_keeps_primary_behavior(monkeypatch):
    original = LaplaceGuidanceTrackingMPC._evaluate_trajectory
    calls = []

    def evaluate(self, *args, **kwargs):
        if kwargs.get("require_feasible"):
            calls.append(1)
            return None
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LaplaceGuidanceTrackingMPC, "_evaluate_trajectory", evaluate)
    m = LaplaceGuidanceTrackingMPC(
        make_filter(),
        config(),
        guidance_config=LaplaceGuidanceTrackingConfig(weight=10),
    )
    r = m.plan(np.zeros(3), (0.2, 0.0), nominal_goal_grid_xy=(2.0, 0.0), **field())
    assert len(calls) == 4 and not r.guidance_tracking.active
    assert r.guidance_tracking.reason == "no_exact_safe_progress_reference"
    assert r.first_command == pytest.approx((0.2, 0.0))


@pytest.mark.parametrize("kwargs", [field(h=2.0), field(g=(0.0, 0.0))])
def test_far_or_zero_guidance_does_not_force_a_reference(kwargs):
    m = LaplaceGuidanceTrackingMPC(
        make_filter(),
        config(),
        guidance_config=LaplaceGuidanceTrackingConfig(weight=10),
    )
    r = m.plan(np.zeros(3), (0.2, 0.0), nominal_goal_grid_xy=(2.0, 0.0), **kwargs)
    assert r.success and not r.guidance_tracking.active


def test_stationary_nominal_and_unsafe_initial_state_remain_stopped():
    m = LaplaceGuidanceTrackingMPC(
        make_filter(),
        config(),
        guidance_config=LaplaceGuidanceTrackingConfig(weight=10),
    )
    r = m.plan(np.zeros(3), (0.0, 0.0), nominal_goal_grid_xy=(2.0, 0.0), **field())
    assert (
        r.success and r.first_command == (0.0, 0.0) and not r.guidance_tracking.active
    )
    r = m.plan(
        np.zeros(3), (0.2, 0.0), nominal_goal_grid_xy=(2.0, 0.0), **field(h=-0.2)
    )
    assert not r.success and r.first_command == (0.0, 0.0)


@pytest.mark.parametrize("goal", [None, [np.nan, 0], [1, 2, 3]])
def test_active_goal_is_required_and_finite(goal):
    with pytest.raises(ValueError, match="nominal_goal_grid_xy"):
        LaplaceGuidanceTrackingMPC(
            make_filter(),
            config(),
            guidance_config=LaplaceGuidanceTrackingConfig(weight=10),
        ).plan(np.zeros(3), (0.2, 0.0), nominal_goal_grid_xy=goal, **field())


def test_does_not_replace_an_unrelated_path_target():
    with pytest.raises(ValueError, match="state-path target"):
        LaplaceGuidanceTrackingMPC(
            make_filter(),
            config(),
            guidance_config=LaplaceGuidanceTrackingConfig(weight=10),
        ).plan(
            np.zeros(3),
            (0.2, 0.0),
            nominal_goal_grid_xy=(2.0, 0.0),
            target_chassis_states_grid=np.zeros((9, 3)),
            **field(),
        )


def test_local_reference_stops_at_its_goal_without_changing_acceptance_radius():
    m = LaplaceGuidanceTrackingMPC(
        make_filter(),
        config(),
        guidance_config=LaplaceGuidanceTrackingConfig(weight=10),
    )
    r = m.plan(
        np.zeros(3), (0.2, 0.0), nominal_goal_grid_xy=(0.1, 0.0), **field(g=(1.0, 0.0))
    )
    d = r.guidance_tracking
    assert d.active
    assert (
        np.linalg.norm(np.array(d.reference_chassis_states[-1][:2]) - [0.1, 0.0])
        <= m.guidance_config.reference_goal_radius_m
    )
    np.testing.assert_array_equal(d.reference_commands[-1], (0.0, 0.0))


def test_runtime_forwards_explicit_goal_and_preserves_epoch():
    from test_safe_sage_runtime import make_runtime, make_observation, make_provider
    from robot.safety.safe_sage_runtime import SafeSAGEControlRuntime
    from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter

    p = make_runtime()
    p.initialize(make_observation(1, 1.0), make_provider())
    paper = make_filter()
    primary = SafeSAGEControlRuntime(
        mpc=SafeSAGEUnicycleMPC(paper, config()),
        backup_filter=SafeSAGEUnicycleCBFFilter(paper),
    )
    with pytest.raises(ValueError, match="explicit guidance ablation"):
        primary.plan(np.zeros(3), (0.2, 0.0), p.field, nominal_goal_grid_xy=(1.0, 0.0))
    c = SafeSAGEControlRuntime(
        mpc=LaplaceGuidanceTrackingMPC(
            paper, config(), guidance_config=LaplaceGuidanceTrackingConfig(weight=10)
        ),
        backup_filter=SafeSAGEUnicycleCBFFilter(paper),
    )
    r = c.plan(np.zeros(3), (0.2, 0.0), p.field, nominal_goal_grid_xy=(1.0, 0.0))
    assert r.guidance_tracking.active_nominal_goal_grid_xy == (1.0, 0.0)
    assert p.field.sensor_timestamp_seconds == 1.0


def test_reference_cost_gradient_qp_and_owned_inputs_are_consistent():
    c, paper = config(), make_filter()
    commands = np.tile([0.15, 0.1], (c.horizon_steps, 1))
    nominal = np.tile([0.2, 0.0], (c.horizon_steps, 1))
    reference = np.tile([0.18, 0.3], (c.horizon_steps, 1))
    weights = np.array([12.0, 3.0])
    m = _ReferenceTrackingMPC(paper, c, reference, weights)
    reference[:] = 99
    weights[:] = 99
    assert not m.reference_commands.flags.writeable
    assert not m.reference_weights.flags.writeable
    trajectory = m._evaluate_trajectory(
        np.zeros(3), commands, temporal_safety_derivative=0.0, **field()
    )
    target = trajectory.chassis_states.copy()
    target[1:, 1] += 0.03
    gradient = m._objective_gradient(
        commands,
        nominal,
        chassis_states=trajectory.chassis_states,
        target_chassis_states=target,
    )
    numerical = np.zeros_like(commands)
    eps = 1e-6
    for i in range(commands.size):
        values = []
        for sign in (-1, 1):
            varied = commands.copy()
            varied.flat[i] += sign * eps
            t = m._evaluate_trajectory(
                np.zeros(3), varied, temporal_safety_derivative=0.0, **field()
            )
            values.append(
                m._objective_value(
                    varied,
                    nominal,
                    chassis_states=t.chassis_states,
                    target_chassis_states=target,
                )
            )
        numerical.flat[i] = (values[1] - values[0]) / (2 * eps)
    np.testing.assert_allclose(gradient, numerical, atol=1e-7)
    kwargs = dict(
        reference=trajectory,
        state_sensitivities=m._rollout_state_sensitivities(
            trajectory.chassis_states, commands
        ),
        target_chassis_states=target,
        slack_variable_count=2,
    )
    p, q = m._quadratic_objective(nominal, **kwargs)
    base_p, base_q = SafeSAGEUnicycleMPC(paper, c)._quadratic_objective(
        nominal, **kwargs
    )
    expected_weights = np.tile([12.0, 3.0], c.horizon_steps)
    np.testing.assert_allclose(
        (p - base_p).toarray(), np.diag(np.r_[2 * expected_weights, 0, 0]), atol=1e-12
    )
    np.testing.assert_allclose(
        q - base_q,
        np.r_[-2 * expected_weights * m.reference_commands.ravel(), 0, 0],
        atol=1e-12,
    )


def test_guidance_blends_to_active_goal_without_changing_physical_field():
    m = LaplaceGuidanceTrackingMPC(
        make_filter(),
        config(),
        guidance_config=LaplaceGuidanceTrackingConfig(weight=10),
    )
    far = m._desired_heading(np.zeros(3), np.array([2.0, 0.0]), (0.0, 1.0))
    near = m._desired_heading(np.zeros(3), np.array([0.05, 0.0]), (0.0, 1.0))
    assert near == pytest.approx(far * 0.01)
    assert far == pytest.approx(np.pi / 3)


def test_turn_only_reference_cannot_displace_a_safe_forward_primary_plan():
    """Natural numeric reproduction: no monkeypatch or executable-gate changes."""
    c = replace(config(), horizon_steps=30, time_step=.2)
    cs, h, g = make_fields(guidance=(0., 1.), safety_value=.3, yaw_slice_count=16,
        unsafe_yaw_indices=tuple(range(7)) + tuple(range(11, 16)))
    values = dict(configuration_space=cs, h_full=h, guidance_fields=g)
    initial = np.zeros(3)
    legacy = LaplaceGuidanceTrackingMPC(make_filter(), c,
        guidance_config=LaplaceGuidanceTrackingConfig(weight=100, minimum_reference_progress_m=0,
            minimum_turn_first_rate_cap=.1))
    old = legacy.plan(initial, (.05, 0.), nominal_goal_grid_xy=(2., 0.), **values)
    assert old.success and old.guidance_tracking.active
    assert old.guidance_tracking.reference_goal_progress_m == pytest.approx(0.)
    assert old.linear_velocity < .001
    actual = LaplaceGuidanceTrackingMPC(make_filter(), c,
        guidance_config=LaplaceGuidanceTrackingConfig(weight=100, minimum_turn_first_rate_cap=.1))
    result = actual.plan(initial, (.05, 0.), nominal_goal_grid_xy=(2., 0.), **values)
    primary = SafeSAGEUnicycleMPC(make_filter(), c).plan(initial, (.05, 0.), **values)
    assert result.success and not result.guidance_tracking.active
    assert result.guidance_tracking.rejected_no_progress_reference_count >= 1
    assert result.first_command == pytest.approx(primary.first_command)
    assert result.linear_velocity == pytest.approx(.05)
    assert result.minimum_dtcbf_margin >= -c.constraint_tolerance


def test_turn_first_with_later_translation_remains_an_eligible_reference():
    m = LaplaceGuidanceTrackingMPC(make_filter(), replace(config(), horizon_steps=30),
        guidance_config=LaplaceGuidanceTrackingConfig(weight=100))
    initial, goal = np.zeros(3), np.array([2., 0.])
    commands = m._guidance_rollout(initial, goal, .2, .8, True, field())
    assert commands[0, 0] == 0.
    states = m._rollout(initial, commands)
    assert np.linalg.norm(goal) - np.linalg.norm(goal-states[-1, :2]) > .01
    checked = m._evaluate_trajectory(initial, commands, temporal_safety_derivative=0., **field())
    assert m._trajectory_is_safe(checked)


@pytest.mark.parametrize("value", [-.01, np.nan, np.inf, True])
def test_invalid_reference_progress_rejected(value):
    with pytest.raises(ValueError):
        LaplaceGuidanceTrackingConfig(minimum_reference_progress_m=value)


@pytest.mark.parametrize("value", [0, -.1, .9, np.nan, True])
def test_invalid_turn_first_pool_cap_rejected(value):
    with pytest.raises(ValueError):
        LaplaceGuidanceTrackingConfig(minimum_turn_first_rate_cap=value)


def test_weak_turn_first_candidates_are_excluded_without_forcing_executable_turns(monkeypatch):
    m = LaplaceGuidanceTrackingMPC(make_filter(), config(),
        guidance_config=LaplaceGuidanceTrackingConfig(weight=100))
    original = m._guidance_rollout
    choices = []
    def record(initial, goal, speed, cap, turn_first, field):
        choices.append((cap, turn_first))
        return original(initial, goal, speed, cap, turn_first, field)
    monkeypatch.setattr(m, "_guidance_rollout", record)
    r = m.plan(np.zeros(3), (.2, 0.), nominal_goal_grid_xy=(2., 0.), **field())
    assert choices == [(.4, False), (.8, False), (.4, True), (.8, True)]
    assert r.success
    assert r.minimum_dtcbf_margin >= -m.config.constraint_tolerance
    # Near-zero requested turns remain possible in a genuinely aligned plan.
    r = m.plan(np.zeros(3), (.2, 0.), nominal_goal_grid_xy=(2., 0.), **field(g=(1., 0.)))
    assert r.linear_velocity > .1 and abs(r.angular_velocity) < .01
