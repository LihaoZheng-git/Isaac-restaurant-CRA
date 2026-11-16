"""Approved route guides are pose-only; final arrival keeps its original gate."""

from dataclasses import replace
import math

import pytest

from robot.navigation.waypoint_controller import (
    RobotPose2D,
    Waypoint,
    WaypointController,
    WaypointControllerConfig,
)


def controller(*, route=None, start=Waypoint(0, 0), **config):
    return WaypointController(
        route or [Waypoint(2.2, 0), Waypoint(3.1, 0), Waypoint(3.9, 0)],
        route_start=start,
        config=WaypointControllerConfig(
            intermediate_waypoints_as_guides=True, **config
        ),
    )


def test_intermediate_plane_can_be_passed_off_route_but_final_cannot():
    guide = controller()
    command = guide.compute_command(RobotPose2D(2.3, 0.45, 0))
    assert command.waypoint_advanced and guide.current_waypoint_index == 1
    assert command.target_waypoint == Waypoint(3.3, 0)
    command = guide.compute_command(RobotPose2D(4.0, 0.45, 0))
    assert guide.current_waypoint_index == 2 and not command.route_complete
    command = guide.compute_command(RobotPose2D(3.85, 0.02, 0))
    assert command.route_complete
    assert (command.linear_velocity, command.angular_velocity) == (0, 0)


def test_default_still_requires_reaching_each_checkpoint():
    guide = controller()
    legacy = WaypointController(guide.waypoints)
    pose = RobotPose2D(2.3, 0.45, 0)
    legacy.compute_command(pose)
    assert legacy.current_waypoint_index == 0
    assert legacy.current_waypoint == Waypoint(2.2, 0)


def test_lookahead_crosses_nearby_guides_without_demanding_a_return():
    guide = controller()
    command = guide.compute_command(RobotPose2D(1.8, -0.4, 0))
    assert guide.current_waypoint_index == 0  # Physical plane not passed yet.
    assert command.target_waypoint == Waypoint(2.8, 0)
    assert command.linear_velocity > 0


def test_far_lateral_pose_does_not_claim_guide_progress_or_arrival():
    guide = controller()
    command = guide.compute_command(RobotPose2D(4, 2.0, 0))
    assert guide.current_waypoint_index == 0
    assert not command.waypoint_advanced and not command.route_complete


def test_polyline_lookahead_and_progress_follow_a_real_route_turn():
    guide = controller(route=[Waypoint(1, 0), Waypoint(1, 2), Waypoint(3, 2)])
    command = guide.compute_command(RobotPose2D(0.8, -0.2, 0))
    assert command.target_waypoint.x == pytest.approx(1)
    assert command.target_waypoint.y == pytest.approx(0.8)
    guide.compute_command(RobotPose2D(1.2, 0.4, math.pi / 2))
    assert guide.current_waypoint_index == 1
    command = guide.compute_command(RobotPose2D(1.2, 2.1, 0))
    assert guide.current_waypoint_index == 2
    assert not command.route_complete


def test_prediction_copies_original_anchor_and_never_commits_progress():
    guide = controller()
    guide.compute_command(RobotPose2D(2.0, -0.4, 0))
    predicted = guide.predict_commands(
        RobotPose2D(2.3, -0.4, 0), horizon_steps=4, time_step=0.2
    )
    assert predicted[0].target_waypoint_index == 1
    assert guide.current_waypoint_index == 0
    assert guide.current_waypoint == Waypoint(2.2, 0)


def test_reset_and_auto_anchor_preserve_pose_only_start_contract():
    guide = controller(start=None, route=[Waypoint(2, 0), Waypoint(3, 0)])
    guide.compute_command(RobotPose2D(0, 0, 0))
    guide.compute_command(RobotPose2D(3, 0, 0))
    assert guide.route_complete
    guide.reset()
    assert guide.current_waypoint_index == 0
    assert not guide.route_complete
    assert guide.compute_command(RobotPose2D(0, 0, 0)).target_waypoint == Waypoint(1, 0)


def test_zero_length_intermediate_segment_is_not_a_zero_distance_final_goal():
    guide = controller(route=[Waypoint(0, 0), Waypoint(0, 0), Waypoint(2, 0)])
    command = guide.compute_command(RobotPose2D(0, 0.4, 0))
    assert guide.current_waypoint_index == 2 and not command.route_complete


def test_single_point_route_never_becomes_a_passed_guide():
    guide = controller(route=[Waypoint(1, 0)])
    assert not guide.compute_command(RobotPose2D(2, 0.4, 0)).route_complete


@pytest.mark.parametrize(
    "key,value",
    [("route_guide_lookahead_m", 0), ("route_guide_max_cross_track_m", math.inf)],
)
def test_invalid_guide_distances_rejected(key, value):
    with pytest.raises(ValueError):
        replace(WaypointControllerConfig(), **{key: value})


def test_guide_mode_must_be_explicit_boolean_and_start_a_waypoint():
    with pytest.raises(TypeError):
        WaypointControllerConfig(intermediate_waypoints_as_guides="yes")
    with pytest.raises(TypeError):
        controller(start=(0, 0))
