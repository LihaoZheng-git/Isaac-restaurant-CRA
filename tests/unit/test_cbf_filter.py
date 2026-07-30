import numpy as np
import pytest

from robot.safety.cbf_filter import (
    CBFConfig,
    CBFSafetyFilter,
)


def make_filter(
    *,
    minimum_velocity: float = 0.0,
) -> CBFSafetyFilter:
    return CBFSafetyFilter(
        CBFConfig(
            safety_radius=0.8,
            alpha=1.0,
            minimum_linear_velocity=minimum_velocity,
            maximum_linear_velocity=0.5,
        )
    )


def test_no_obstacles_preserves_nominal_command() -> None:
    safety_filter = make_filter()

    result = safety_filter.filter_command(
        nominal_linear_velocity=0.2,
        nominal_angular_velocity=0.1,
        obstacle_points_xy=np.empty((0, 2)),
    )

    assert result.safe_linear_velocity == pytest.approx(0.2)
    assert result.safe_angular_velocity == pytest.approx(0.1)
    assert result.obstacle_count == 0
    assert result.constraint_feasible
    assert not result.intervention_active
    assert not result.emergency_stop_active
    assert np.isinf(result.minimum_obstacle_distance)
    assert np.isinf(result.minimum_barrier_value)


def test_far_obstacle_does_not_change_command() -> None:
    safety_filter = make_filter()

    result = safety_filter.filter_command(
        nominal_linear_velocity=0.2,
        nominal_angular_velocity=0.0,
        obstacle_points_xy=np.array([[2.0, 0.0]]),
    )

    assert result.safe_linear_velocity == pytest.approx(0.2)
    assert not result.intervention_active


def test_near_front_obstacle_reduces_velocity() -> None:
    safety_filter = make_filter()

    result = safety_filter.filter_command(
        nominal_linear_velocity=0.25,
        nominal_angular_velocity=0.1,
        obstacle_points_xy=np.array([[0.9, 0.0]]),
    )

    expected_barrier = 0.9**2 - 0.8**2
    expected_limit = expected_barrier / (2.0 * 0.9)

    assert result.minimum_obstacle_distance == pytest.approx(0.9)
    assert result.minimum_barrier_value == pytest.approx(
        expected_barrier
    )
    assert result.safe_linear_velocity == pytest.approx(
        expected_limit
    )
    assert result.safe_angular_velocity == pytest.approx(0.1)
    assert result.intervention_active
    assert result.constraint_feasible
    assert not result.emergency_stop_active


def test_obstacle_at_safety_boundary_stops_robot() -> None:
    safety_filter = make_filter()

    result = safety_filter.filter_command(
        nominal_linear_velocity=0.2,
        nominal_angular_velocity=0.0,
        obstacle_points_xy=np.array([[0.8, 0.0]]),
    )

    assert result.minimum_barrier_value == pytest.approx(0.0)
    assert result.safe_linear_velocity == pytest.approx(0.0)
    assert result.intervention_active
    assert result.constraint_feasible


def test_obstacle_behind_does_not_limit_forward_motion() -> None:
    safety_filter = make_filter()

    result = safety_filter.filter_command(
        nominal_linear_velocity=0.2,
        nominal_angular_velocity=0.0,
        obstacle_points_xy=np.array([[-0.9, 0.0]]),
    )

    assert result.safe_linear_velocity == pytest.approx(0.2)
    assert not result.intervention_active


def test_nearest_front_constraint_is_applied() -> None:
    safety_filter = make_filter()

    result = safety_filter.filter_command(
        nominal_linear_velocity=0.25,
        nominal_angular_velocity=0.0,
        obstacle_points_xy=np.array(
            [
                [2.0, 0.0],
                [1.4, 0.2],
                [0.9, 0.0],
            ]
        ),
    )

    expected_limit = (
        0.9**2 - 0.8**2
    ) / (2.0 * 0.9)

    assert result.safe_linear_velocity == pytest.approx(
        expected_limit
    )
    assert result.obstacle_count == 3
    assert result.intervention_active


def test_unsafe_initial_state_uses_emergency_stop() -> None:
    safety_filter = make_filter()

    result = safety_filter.filter_command(
        nominal_linear_velocity=0.2,
        nominal_angular_velocity=0.0,
        obstacle_points_xy=np.array([[0.5, 0.0]]),
    )

    assert result.minimum_barrier_value < 0.0
    assert result.safe_linear_velocity == pytest.approx(0.0)
    assert not result.constraint_feasible
    assert result.intervention_active
    assert result.emergency_stop_active


def test_invalid_obstacle_shape_is_rejected() -> None:
    safety_filter = make_filter()

    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        safety_filter.filter_command(
            nominal_linear_velocity=0.2,
            nominal_angular_velocity=0.0,
            obstacle_points_xy=np.zeros((4, 3)),
        )


def test_nonfinite_obstacle_point_is_rejected() -> None:
    safety_filter = make_filter()

    with pytest.raises(ValueError, match="finite"):
        safety_filter.filter_command(
            nominal_linear_velocity=0.2,
            nominal_angular_velocity=0.0,
            obstacle_points_xy=np.array([[np.nan, 0.0]]),
        )


@pytest.mark.parametrize(
    "safety_radius, alpha",
    [
        (0.0, 1.0),
        (-0.5, 1.0),
        (0.8, 0.0),
        (0.8, -1.0),
    ],
)
def test_invalid_cbf_configuration_is_rejected(
    safety_radius: float,
    alpha: float,
) -> None:
    with pytest.raises(ValueError):
        CBFConfig(
            safety_radius=safety_radius,
            alpha=alpha,
        )


def test_omitted_velocities_equal_explicit_zero_velocities() -> None:
    safety_filter = make_filter()
    points = np.array([[1.0, 0.0], [1.5, 0.2]])

    omitted_result = safety_filter.filter_command(
        nominal_linear_velocity=0.25,
        nominal_angular_velocity=0.1,
        obstacle_points_xy=points,
    )

    zero_result = safety_filter.filter_command(
        nominal_linear_velocity=0.25,
        nominal_angular_velocity=0.1,
        obstacle_points_xy=points,
        obstacle_velocities_xy=np.zeros_like(points),
    )

    assert omitted_result == zero_result


def test_approaching_obstacle_reduces_velocity_earlier() -> None:
    safety_filter = make_filter()

    result = safety_filter.filter_command(
        nominal_linear_velocity=0.25,
        nominal_angular_velocity=0.0,
        obstacle_points_xy=np.array([[1.0, 0.0]]),
        obstacle_velocities_xy=np.array([[-0.1, 0.0]]),
    )

    barrier = 1.0**2 - 0.8**2
    expected_limit = (
        2.0 * 1.0 * -0.1 + barrier
    ) / (2.0 * 1.0)

    assert result.safe_linear_velocity == pytest.approx(
        expected_limit
    )
    assert result.safe_linear_velocity == pytest.approx(0.08)
    assert result.intervention_active
    assert result.constraint_feasible
    assert not result.emergency_stop_active


def test_receding_obstacle_allows_more_forward_motion() -> None:
    safety_filter = make_filter()
    points = np.array([[1.0, 0.0]])

    stationary_result = safety_filter.filter_command(
        nominal_linear_velocity=0.25,
        nominal_angular_velocity=0.0,
        obstacle_points_xy=points,
    )

    receding_result = safety_filter.filter_command(
        nominal_linear_velocity=0.25,
        nominal_angular_velocity=0.0,
        obstacle_points_xy=points,
        obstacle_velocities_xy=np.array([[0.1, 0.0]]),
    )

    assert (
        receding_result.admissible_linear_velocity_maximum
        > stationary_result.admissible_linear_velocity_maximum
    )
    assert receding_result.safe_linear_velocity == pytest.approx(0.25)
    assert not receding_result.intervention_active


def test_approaching_lateral_obstacle_can_be_infeasible() -> None:
    safety_filter = make_filter()

    result = safety_filter.filter_command(
        nominal_linear_velocity=0.2,
        nominal_angular_velocity=0.0,
        obstacle_points_xy=np.array([[0.0, 0.9]]),
        obstacle_velocities_xy=np.array([[0.0, -0.2]]),
    )

    assert not result.constraint_feasible
    assert result.safe_linear_velocity == pytest.approx(0.0)
    assert result.intervention_active
    assert result.emergency_stop_active


def test_mismatched_obstacle_velocity_shape_is_rejected() -> None:
    safety_filter = make_filter()

    with pytest.raises(ValueError, match="same shape"):
        safety_filter.filter_command(
            nominal_linear_velocity=0.2,
            nominal_angular_velocity=0.0,
            obstacle_points_xy=np.zeros((2, 2)),
            obstacle_velocities_xy=np.zeros((1, 2)),
        )


def test_nonfinite_obstacle_velocity_is_rejected() -> None:
    safety_filter = make_filter()

    with pytest.raises(ValueError, match="finite"):
        safety_filter.filter_command(
            nominal_linear_velocity=0.2,
            nominal_angular_velocity=0.0,
            obstacle_points_xy=np.array([[1.0, 0.0]]),
            obstacle_velocities_xy=np.array([[np.nan, 0.0]]),
        )


def test_explicit_empty_obstacle_velocities_are_valid() -> None:
    safety_filter = make_filter()

    result = safety_filter.filter_command(
        nominal_linear_velocity=0.2,
        nominal_angular_velocity=0.0,
        obstacle_points_xy=np.empty((0, 2)),
        obstacle_velocities_xy=np.empty((0, 2)),
    )

    assert result.obstacle_count == 0
    assert result.safe_linear_velocity == pytest.approx(0.2)
    assert result.constraint_feasible