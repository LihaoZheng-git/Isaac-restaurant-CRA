from dataclasses import fields

import numpy as np
import pytest
from test_safe_sage_cbf_filter import make_fields, make_filter

from robot.safety.laplace_guidance_tracking_mpc import (
    LaplaceGuidanceTrackingConfig,
    LaplaceGuidanceTrackingMPC,
    _GuidanceObjectiveMPC,
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


def test_zero_weight_delegates_identically_to_primary():
    paper = make_filter()
    c = config()
    primary = SafeSAGEUnicycleMPC(paper, c).plan(np.zeros(3), (0.2, 0.0), **field())
    ablation = LaplaceGuidanceTrackingMPC(paper, c).plan(
        np.zeros(3), (0.2, 0.0), **field()
    )
    assert type(primary) is type(ablation)
    for f in fields(primary):
        a, b = getattr(primary, f.name), getattr(ablation, f.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_explicit_objective_tracks_guidance_when_primary_constraint_is_inactive(side):
    paper, c = make_filter(), config()
    kwargs = field(g=(1.0, side))
    nominal = np.tile([0.2, 0.0], (c.horizon_steps, 1))
    original = nominal.copy()
    primary = SafeSAGEUnicycleMPC(paper, c).plan(np.zeros(3), nominal, **kwargs)
    ablation = LaplaceGuidanceTrackingMPC(
        paper, c, guidance_config=LaplaceGuidanceTrackingConfig(weight=100.0)
    ).plan(np.zeros(3), nominal, **kwargs)
    np.testing.assert_array_equal(nominal, original)
    assert primary.success and ablation.success
    assert abs(primary.angular_velocity) < 1e-8
    assert side * ablation.angular_velocity > 0.05
    assert side * ablation.predicted_chassis_states[-1, 1] > 0.001
    assert ablation.minimum_dtcbf_margin >= -c.constraint_tolerance
    assert ablation.guidance_tracking.nominal_unchanged
    assert max(ablation.guidance_tracking.activation) > 0
    assert all(
        abs(yaw) <= np.pi
        for yaw in ablation.guidance_tracking.heading_reference_radians
    )
    d = ablation.guidance_tracking
    errors = (
        np.array(d.heading_reference_radians)
        - d.nominal_terminal_heading_radians
        + np.pi
    ) % (2 * np.pi) - np.pi
    assert (
        np.max(np.abs(errors)[np.array(d.activation) > 0])
        <= d.maximum_heading_bias_radians + 1e-12
    )


def test_far_field_and_zero_guidance_do_not_create_an_objective():
    for kwargs in (field(h=2.0), field(g=(0.0, 0.0))):
        c = config()
        result = LaplaceGuidanceTrackingMPC(
            make_filter(),
            c,
            guidance_config=LaplaceGuidanceTrackingConfig(weight=100.0),
        ).plan(np.zeros(3), (0.2, 0.0), **kwargs)
        assert result.success
        assert result.first_command == pytest.approx((0.2, 0.0), abs=1e-8)
        assert max(result.guidance_tracking.activation) == 0


def test_unsafe_initial_state_still_stops():
    result = LaplaceGuidanceTrackingMPC(
        make_filter(),
        config(),
        guidance_config=LaplaceGuidanceTrackingConfig(weight=100.0),
    ).plan(np.zeros(3), (0.2, 0.0), **field(h=-0.2))
    assert not result.success
    assert result.first_command == (0.0, 0.0)
    assert result.status == "negative_safety_value"


def test_guidance_heading_is_covariant_under_field_frame_rotation():
    outputs = []
    for yaw in (0.0, np.pi / 2, -np.pi + 0.1):
        direction = yaw + np.pi / 4
        result = LaplaceGuidanceTrackingMPC(
            make_filter(),
            config(),
            guidance_config=LaplaceGuidanceTrackingConfig(weight=10.0),
        ).plan(
            np.array([0.0, 0.0, yaw]),
            (0.2, 0.0),
            **field(g=(np.cos(direction), np.sin(direction))),
        )
        assert result.success
        outputs.append(result.first_command)
    np.testing.assert_allclose(outputs, [outputs[0]] * len(outputs), atol=1e-5)


def test_guidance_does_not_invent_motion_at_a_stationary_nominal():
    result = LaplaceGuidanceTrackingMPC(
        make_filter(),
        config(),
        guidance_config=LaplaceGuidanceTrackingConfig(weight=10.0),
    ).plan(np.zeros(3), (0.0, 0.0), **field())
    assert result.success
    assert result.first_command == (0.0, 0.0)
    assert max(result.guidance_tracking.activation) == 0


def test_rejected_guidance_seed_is_never_used(monkeypatch):
    original = _GuidanceObjectiveMPC._evaluate_trajectory
    rejected = []

    def evaluate(self, *args, **kwargs):
        if kwargs.get("require_feasible"):
            rejected.append(1)
            return None
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_GuidanceObjectiveMPC, "_evaluate_trajectory", evaluate)
    result = LaplaceGuidanceTrackingMPC(
        make_filter(),
        config(),
        guidance_config=LaplaceGuidanceTrackingConfig(weight=10.0),
    ).plan(np.zeros(3), (0.2, 0.0), **field())
    assert len(rejected) == 2
    assert not result.guidance_tracking.guidance_seed_used
    assert (
        result.success and result.minimum_dtcbf_margin >= -config().constraint_tolerance
    )


def test_guidance_validates_optional_nominal_path_before_cost_evaluation():
    with pytest.raises(ValueError, match="target_chassis_states_grid"):
        LaplaceGuidanceTrackingMPC(
            make_filter(),
            config(),
            guidance_config=LaplaceGuidanceTrackingConfig(weight=10.0),
        ).plan(
            np.zeros(3),
            (0.2, 0.0),
            target_chassis_states_grid=np.zeros((2, 3)),
            **field(),
        )


def test_additive_objective_gradient_and_qp_agree():
    c = config()
    rng = np.random.default_rng(4)
    targets = rng.uniform(-0.2, 0.2, c.horizon_steps)
    weights = rng.uniform(0.1, 0.9, targets.shape)
    worker = _GuidanceObjectiveMPC(make_filter(), c, targets=targets, weights=weights)
    targets[:] = 99  # the per-plan cost owns its data
    assert np.max(worker.targets) < 1
    commands = rng.uniform(0.01, 0.2, (c.horizon_steps, 2))
    nominal = np.zeros_like(commands)
    states = worker._rollout(np.zeros(3), commands)
    assert worker._objective_value(commands, nominal) <= worker._objective_value(
        commands, nominal, chassis_states=states
    )
    analytic = worker._objective_gradient(
        commands, nominal, chassis_states=states, target_chassis_states=None
    )
    epsilon = 1e-6
    numerical = np.zeros_like(commands)
    for i in range(commands.size):
        plus, minus = commands.copy(), commands.copy()
        plus.flat[i] += epsilon
        minus.flat[i] -= epsilon
        numerical.flat[i] = (
            worker._objective_value(
                plus, nominal, chassis_states=worker._rollout(np.zeros(3), plus)
            )
            - worker._objective_value(
                minus, nominal, chassis_states=worker._rollout(np.zeros(3), minus)
            )
        ) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numerical, atol=1e-9)
    from types import SimpleNamespace

    p, q = worker._quadratic_objective(
        nominal,
        reference=SimpleNamespace(commands=commands, chassis_states=states),
        state_sensitivities=worker._rollout_state_sensitivities(states, commands),
        target_chassis_states=None,
        slack_variable_count=2,
    )
    upper = p.toarray()[: commands.size, : commands.size]
    full = upper + upper.T - np.diag(np.diag(upper))
    np.testing.assert_allclose(
        full @ commands.reshape(-1) + q[: commands.size],
        analytic.reshape(-1),
        atol=1e-9,
    )
