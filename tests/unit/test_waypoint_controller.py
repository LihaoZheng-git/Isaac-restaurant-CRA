"""Unit tests for the pure waypoint controller."""

import math

import pytest

from robot.navigation.waypoint_controller import (
    RobotPose2D,
    Waypoint,
    WaypointController,
    WaypointControllerConfig,
    WaypointMode,
    wrap_to_pi,
)


def test_wrap_to_pi_preserves_zero() -> None:
    """Zero should remain zero."""

    assert wrap_to_pi(0.0) == pytest.approx(0.0)


def test_wrap_to_pi_handles_full_rotations() -> None:
    """Complete rotations should wrap back to zero."""

    assert wrap_to_pi(
        2.0 * math.pi
    ) == pytest.approx(0.0)

    assert wrap_to_pi(
        -2.0 * math.pi
    ) == pytest.approx(0.0)


def test_wrap_to_pi_uses_shortest_boundary_direction() -> None:
    """Crossing the pi boundary should produce a small error."""

    desired_heading = math.radians(-179.0)
    current_heading = math.radians(179.0)

    heading_error = wrap_to_pi(
        desired_heading - current_heading
    )

    assert heading_error == pytest.approx(
        math.radians(2.0)
    )


def test_controller_drives_straight_toward_waypoint() -> None:
    """An aligned robot should drive directly forward."""

    controller = WaypointController(
        [
            Waypoint(1.0, 0.0),
        ]
    )

    command = controller.compute_command(
        RobotPose2D(
            x=0.0,
            y=0.0,
            yaw=0.0,
        )
    )

    assert command.mode is WaypointMode.DRIVING
    assert command.linear_velocity == pytest.approx(
        0.20
    )
    assert command.angular_velocity == pytest.approx(
        0.0
    )
    assert command.distance_to_waypoint == pytest.approx(
        1.0
    )
    assert command.heading_error == pytest.approx(
        0.0
    )
    assert command.target_waypoint_index == 0
    assert not command.route_complete


@pytest.mark.parametrize(
    ("target", "expected_angular_velocity"),
    [
        (Waypoint(0.0, 1.0), 0.60),
        (Waypoint(0.0, -1.0), -0.60),
    ],
)
def test_controller_rotates_for_large_heading_error(
    target: Waypoint,
    expected_angular_velocity: float,
) -> None:
    """Large left and right errors should rotate in place."""

    controller = WaypointController([target])

    command = controller.compute_command(
        RobotPose2D(
            x=0.0,
            y=0.0,
            yaw=0.0,
        )
    )

    assert command.mode is WaypointMode.ROTATING
    assert command.linear_velocity == pytest.approx(
        0.0
    )
    assert command.angular_velocity == pytest.approx(
        expected_angular_velocity
    )


def test_controller_slows_near_waypoint() -> None:
    """The proportional gain should reduce speed near a target."""

    controller = WaypointController(
        [
            Waypoint(0.20, 0.0),
        ]
    )

    command = controller.compute_command(
        RobotPose2D(
            x=0.0,
            y=0.0,
            yaw=0.0,
        )
    )

    expected_velocity = 0.80 * 0.20

    assert command.mode is WaypointMode.DRIVING
    assert command.linear_velocity == pytest.approx(
        expected_velocity
    )
    assert (
        command.linear_velocity
        < controller.config.maximum_linear_velocity
    )


def test_configured_alignment_floor_preserves_obstacle_unaware_progress() -> None:
    controller = WaypointController(
        [Waypoint(1.0, 0.0)],
        config=WaypointControllerConfig(
            minimum_alignment_scale=0.25,
            rotate_in_place_threshold_radians=math.pi,
        ),
    )

    command = controller.compute_command(
        RobotPose2D(x=0.0, y=0.0, yaw=math.pi / 2.0)
    )

    assert command.mode is WaypointMode.DRIVING
    assert command.linear_velocity == pytest.approx(0.05)
    assert command.angular_velocity == pytest.approx(-0.60)


def test_controller_advances_reached_waypoint() -> None:
    """A reached waypoint should advance to the next target."""

    route = [
        Waypoint(0.05, 0.0),
        Waypoint(1.00, 0.0),
    ]

    controller = WaypointController(route)

    command = controller.compute_command(
        RobotPose2D(
            x=0.0,
            y=0.0,
            yaw=0.0,
        )
    )

    assert command.waypoint_advanced
    assert command.target_waypoint_index == 1
    assert command.target_waypoint == route[1]
    assert command.mode is WaypointMode.DRIVING
    assert not command.route_complete


def test_controller_stops_after_final_waypoint() -> None:
    """Reaching the final waypoint should complete the route."""

    controller = WaypointController(
        [
            Waypoint(0.05, 0.0),
        ]
    )

    command = controller.compute_command(
        RobotPose2D(
            x=0.0,
            y=0.0,
            yaw=0.0,
        )
    )

    assert command.mode is WaypointMode.COMPLETED
    assert command.linear_velocity == pytest.approx(
        0.0
    )
    assert command.angular_velocity == pytest.approx(
        0.0
    )
    assert command.waypoint_advanced
    assert command.route_complete
    assert controller.route_complete
    assert controller.current_waypoint is None
    assert controller.current_waypoint_index is None


def test_controller_reset_returns_to_first_waypoint() -> None:
    """Reset should make the first waypoint active again."""

    controller = WaypointController(
        [
            Waypoint(0.05, 0.0),
        ]
    )

    controller.compute_command(
        RobotPose2D(
            x=0.0,
            y=0.0,
            yaw=0.0,
        )
    )

    assert controller.route_complete

    controller.reset()

    assert not controller.route_complete
    assert controller.current_waypoint_index == 0
    assert controller.current_waypoint == Waypoint(
        0.05,
        0.0,
    )


def test_prediction_captures_rotation_to_driving_transition() -> None:
    controller = WaypointController([Waypoint(1.0, 0.0)])

    commands = controller.predict_commands(
        RobotPose2D(0.0, 0.0, math.pi / 2.0),
        horizon_steps=8,
        time_step=0.5,
    )

    assert commands[0].mode is WaypointMode.ROTATING
    assert commands[0].linear_velocity == pytest.approx(0.0)
    assert any(command.mode is WaypointMode.DRIVING for command in commands)
    assert any(command.linear_velocity > 0.0 for command in commands)


def test_prediction_does_not_mutate_live_route_state() -> None:
    controller = WaypointController(
        [
            Waypoint(0.05, 0.0),
            Waypoint(1.0, 0.0),
        ]
    )

    commands = controller.predict_commands(
        RobotPose2D(0.0, 0.0, 0.0),
        horizon_steps=4,
        time_step=0.2,
    )

    assert len(commands) == 4
    assert commands[0].target_waypoint == Waypoint(1.0, 0.0)
    assert controller.current_waypoint_index == 0


@pytest.mark.parametrize(
    ("horizon_steps", "time_step"),
    [
        (0, 0.2),
        (2, 0.0),
        (2, math.inf),
    ],
)
def test_prediction_rejects_invalid_configuration(
    horizon_steps: int,
    time_step: float,
) -> None:
    controller = WaypointController([Waypoint(1.0, 0.0)])

    with pytest.raises(ValueError):
        controller.predict_commands(
            RobotPose2D(0.0, 0.0, 0.0),
            horizon_steps=horizon_steps,
            time_step=time_step,
        )


def test_controller_rejects_empty_route() -> None:
    """A controller requires at least one waypoint."""

    with pytest.raises(
        ValueError,
        match="At least one waypoint",
    ):
        WaypointController([])


def test_controller_rejects_non_waypoint_entries() -> None:
    """Raw tuples should not silently enter the route."""

    with pytest.raises(
        TypeError,
        match="Every route entry",
    ):
        WaypointController(
            [
                (1.0, 0.0),  # type: ignore[list-item]
            ]
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("maximum_linear_velocity", 0.0),
        ("maximum_angular_velocity", -1.0),
        ("linear_gain", math.inf),
        ("position_tolerance", 0.0),
        ("minimum_alignment_scale", -0.1),
        ("minimum_alignment_scale", 1.1),
        (
            "rotate_in_place_threshold_radians",
            math.pi + 0.1,
        ),
    ],
)
def test_config_rejects_invalid_values(
    field_name: str,
    invalid_value: float,
) -> None:
    """Invalid configuration values should fail immediately."""

    with pytest.raises(ValueError):
        WaypointControllerConfig(
            **{
                field_name: invalid_value,
            }
        )
