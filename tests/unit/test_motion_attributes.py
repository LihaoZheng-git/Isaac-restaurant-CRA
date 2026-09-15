from dataclasses import FrozenInstanceError
import math

import pytest

from robot.perception.motion_attributes import (
    MotionAttributeConfig,
    RouteMotionContext,
    estimate_track_motion,
)
from robot.perception.physical_attributes import EstimateStatus
from robot.safety.obstacle_tracking import TrackedObstacle


def track(
    *,
    position: tuple[float, float] = (2.0, 1.0),
    velocity: tuple[float, float] = (0.0, -0.5),
    observation_count: int = 6,
    matched: bool = True,
) -> TrackedObstacle:
    return TrackedObstacle(
        track_id=4,
        centroid_world_xy=position,
        centroid_robot_xy=position,
        velocity_world_xy=velocity,
        velocity_robot_xy=velocity,
        point_count=12,
        observation_count=observation_count,
        matched_previous_track=matched,
    )


def route(
    *,
    start: tuple[float, float] = (0.0, 0.0),
    end: tuple[float, float] = (4.0, 0.0),
    speed: float = 0.2,
) -> RouteMotionContext:
    return RouteMotionContext(
        robot_position_world_xy=start,
        waypoint_world_xy=end,
        nominal_robot_speed_mps=speed,
    )


def test_confirmed_tracker_velocity_populates_world_frame_components() -> None:
    motion = estimate_track_motion(
        track(),
        route_context=route(),
        timestamp_seconds=3.0,
    )

    assert motion.velocity_x_mps.value == pytest.approx(0.0)
    assert motion.velocity_y_mps.value == pytest.approx(-0.5)
    assert motion.velocity_x_mps.status is EstimateStatus.INFERRED
    assert "world x" in motion.velocity_x_mps.evidence[0]


def test_relative_speed_uses_waypoint_aligned_nominal_robot_velocity() -> None:
    motion = estimate_track_motion(
        track(),
        route_context=route(),
        timestamp_seconds=3.0,
    )

    assert motion.relative_speed_mps.value == pytest.approx(math.hypot(0.2, 0.5))


def test_relative_speed_prefers_measured_robot_velocity() -> None:
    context = RouteMotionContext(
        robot_position_world_xy=(0, 0),
        waypoint_world_xy=(4, 0),
        nominal_robot_speed_mps=9,
        measured_robot_velocity_world_xy=(0.3, -0.1),
    )
    motion = estimate_track_motion(track(), route_context=context, timestamp_seconds=3)
    assert motion.relative_speed_mps.value == pytest.approx(math.hypot(0.3, 0.4))
    assert "measured robot velocity" in motion.relative_speed_mps.evidence[0]


def test_crossing_prediction_scores_route_center_intersection_high() -> None:
    motion = estimate_track_motion(
        track(),
        route_context=route(),
        timestamp_seconds=3.0,
    )

    assert motion.route_crossing_likelihood.value == pytest.approx(1.0)
    assert motion.route_crossing_likelihood.lower_bound == pytest.approx(0.85)
    assert motion.route_crossing_likelihood.upper_bound == pytest.approx(1.0)


def test_parallel_motion_outside_corridor_scores_zero() -> None:
    motion = estimate_track_motion(
        track(position=(1.0, 1.0), velocity=(0.4, 0.0)),
        route_context=route(),
        timestamp_seconds=3.0,
    )

    assert motion.route_crossing_likelihood.value == pytest.approx(0.0)


def test_static_obstacle_on_route_scores_one() -> None:
    motion = estimate_track_motion(
        track(position=(2.0, 0.0), velocity=(0.0, 0.0)),
        route_context=route(),
        timestamp_seconds=3.0,
    )

    assert motion.route_crossing_likelihood.value == pytest.approx(1.0)


def test_near_route_score_changes_continuously_with_separation() -> None:
    motion = estimate_track_motion(
        track(position=(2.0, 0.25), velocity=(0.0, 0.0)),
        route_context=route(),
        timestamp_seconds=3.0,
    )

    assert motion.route_crossing_likelihood.value == pytest.approx(0.5)


def test_new_track_does_not_publish_false_zero_velocity() -> None:
    motion = estimate_track_motion(
        track(observation_count=1, matched=False),
        route_context=route(),
        timestamp_seconds=3.0,
    )

    assert not motion.velocity_x_mps.is_available
    assert not motion.velocity_y_mps.is_available
    assert not motion.relative_speed_mps.is_available
    assert motion.route_crossing_likelihood.is_available
    assert motion.route_crossing_likelihood.status is EstimateStatus.FALLBACK
    assert motion.route_crossing_likelihood.lower_bound == pytest.approx(0.0)
    assert motion.route_crossing_likelihood.upper_bound == pytest.approx(1.0)


def test_zero_length_route_makes_crossing_score_unavailable() -> None:
    motion = estimate_track_motion(
        track(),
        route_context=route(end=(0.0, 0.0), speed=0.0),
        timestamp_seconds=3.0,
    )

    assert not motion.route_crossing_likelihood.is_available
    assert motion.relative_speed_mps.value == pytest.approx(0.5)


def test_more_observations_increase_but_cap_velocity_confidence() -> None:
    settings = MotionAttributeConfig(
        minimum_observation_count=2,
        full_confidence_observation_count=6,
        minimum_velocity_confidence=0.3,
        maximum_velocity_confidence=0.9,
    )
    early = estimate_track_motion(
        track(observation_count=2),
        route_context=route(),
        timestamp_seconds=2.0,
        config=settings,
    )
    mature = estimate_track_motion(
        track(observation_count=20),
        route_context=route(),
        timestamp_seconds=3.0,
        config=settings,
    )

    assert early.velocity_x_mps.confidence == pytest.approx(0.3)
    assert mature.velocity_x_mps.confidence == pytest.approx(0.9)


def test_route_context_is_immutable_and_copies_vectors() -> None:
    start = [0.0, 0.0]
    context = RouteMotionContext(
        robot_position_world_xy=start,  # type: ignore[arg-type]
        waypoint_world_xy=(1.0, 0.0),
        nominal_robot_speed_mps=0.2,
    )
    start[0] = 10.0

    assert context.robot_position_world_xy == (0.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        context.nominal_robot_speed_mps = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("start", "end", "speed"),
    [
        ((0.0,), (1.0, 0.0), 0.2),
        ((0.0, 0.0), (float("nan"), 0.0), 0.2),
        ((0.0, 0.0), (1.0, 0.0), -0.1),
    ],
)
def test_invalid_route_context_is_rejected(
    start: tuple[float, ...],
    end: tuple[float, ...],
    speed: float,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RouteMotionContext(
            robot_position_world_xy=start,  # type: ignore[arg-type]
            waypoint_world_xy=end,  # type: ignore[arg-type]
            nominal_robot_speed_mps=speed,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"minimum_observation_count": 0},
        {
            "minimum_observation_count": 4,
            "full_confidence_observation_count": 3,
        },
        {"velocity_uncertainty_mps": 0.0},
        {"prediction_horizon_seconds": -1.0},
        {"route_corridor_half_width_m": 0.0},
        {"route_score_uncertainty": 1.1},
        {
            "minimum_velocity_confidence": 0.8,
            "maximum_velocity_confidence": 0.2,
        },
    ],
)
def test_invalid_motion_configuration_is_rejected(
    changes: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        MotionAttributeConfig(**changes)  # type: ignore[arg-type]


def test_negative_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        estimate_track_motion(
            track(),
            route_context=route(),
            timestamp_seconds=-1.0,
        )
