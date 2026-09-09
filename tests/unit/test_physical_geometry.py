import numpy as np
import pytest

from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.physical_attributes import EstimateStatus
from robot.perception.physical_geometry import (
    LidarGeometryConfig,
    estimate_track_geometry,
)
from robot.safety.obstacle_tracking import ObstacleTrackingResult


def tracking_result(
    track_ids: np.ndarray,
    *,
    timestamp: float = 2.0,
) -> ObstacleTrackingResult:
    point_count = int(track_ids.size)
    return ObstacleTrackingResult(
        timestamp_seconds=timestamp,
        point_velocities_robot_xy=np.zeros((point_count, 2)),
        point_track_ids=track_ids,
        tracked_obstacles=(),
        cluster_count=1,
        matched_track_count=1,
        new_track_count=0,
    )


def exact_config(**changes: object) -> LidarGeometryConfig:
    values: dict[str, object] = {
        "minimum_track_points": 4,
        "full_confidence_point_count": 4,
        "lower_quantile": 0.0,
        "upper_quantile": 1.0,
        "range_uncertainty_m": 0.02,
        "unobserved_extent_margin_m": 0.10,
    }
    values.update(changes)
    return LidarGeometryConfig(**values)  # type: ignore[arg-type]


def test_geometry_uses_only_points_assigned_to_requested_track() -> None:
    points = np.array(
        [
            [1.0, -0.4, 0.1],
            [1.3, 0.4, 0.9],
            [1.1, -0.2, 0.3],
            [1.2, 0.2, 0.7],
            [9.0, -4.0, 5.0],
            [9.5, 4.0, 8.0],
        ]
    )
    result = tracking_result(np.array([3, 3, 3, 3, 8, 8]))

    geometry = estimate_track_geometry(
        points,
        tracking_result=result,
        track_id=3,
        floor_height_robot_m=0.0,
        config=exact_config(),
    )

    assert geometry.width_m.value == pytest.approx(0.8)
    assert geometry.depth_m.value == pytest.approx(0.3)
    assert geometry.height_m.value == pytest.approx(0.8)
    assert geometry.center_height_m.value == pytest.approx(0.5)
    assert geometry.fall_height_m.value is None


def test_geometry_bounds_include_configured_uncertainty() -> None:
    points = np.array(
        [
            [1.0, -0.2, 0.1],
            [1.2, 0.2, 0.5],
            [1.1, -0.1, 0.2],
            [1.1, 0.1, 0.4],
        ]
    )

    geometry = estimate_track_geometry(
        points,
        tracking_result=tracking_result(np.full(4, 2)),
        track_id=2,
        floor_height_robot_m=0.0,
        config=exact_config(),
    )

    assert geometry.width_m.lower_bound == pytest.approx(0.38)
    assert geometry.width_m.upper_bound == pytest.approx(0.52)
    assert geometry.width_m.status is EstimateStatus.INFERRED


def test_center_height_is_relative_to_supplied_floor() -> None:
    points = np.array(
        [
            [1.0, -0.2, 0.3],
            [1.2, 0.2, 0.7],
            [1.1, -0.1, 0.4],
            [1.1, 0.1, 0.6],
        ]
    )

    geometry = estimate_track_geometry(
        points,
        tracking_result=tracking_result(np.full(4, 2)),
        track_id=2,
        floor_height_robot_m=0.1,
        config=exact_config(),
    )

    assert geometry.height_m.value == pytest.approx(0.4)
    assert geometry.center_height_m.value == pytest.approx(0.4)


def test_sparse_track_returns_explicitly_unavailable_geometry() -> None:
    points = np.array(
        [
            [1.0, 0.0, 0.2],
            [1.0, 0.1, 0.3],
            [1.0, 0.2, 0.4],
        ]
    )

    geometry = estimate_track_geometry(
        points,
        tracking_result=tracking_result(np.full(3, 4)),
        track_id=4,
        floor_height_robot_m=0.0,
    )

    assert all(
        not estimate.is_available
        for estimate in (
            geometry.width_m,
            geometry.depth_m,
            geometry.height_m,
            geometry.center_height_m,
            geometry.fall_height_m,
        )
    )
    assert "3 aligned LiDAR returns" in geometry.width_m.evidence[0]


def test_absent_track_is_not_converted_to_zero_geometry() -> None:
    points = np.ones((4, 3))

    geometry = estimate_track_geometry(
        points,
        tracking_result=tracking_result(np.full(4, 5)),
        track_id=7,
        floor_height_robot_m=0.0,
    )

    assert geometry.width_m.value is None
    assert geometry.width_m.confidence == pytest.approx(0.0)


def test_point_support_increases_but_caps_geometry_confidence() -> None:
    settings = exact_config(
        full_confidence_point_count=8,
        minimum_confidence=0.3,
        maximum_confidence=0.9,
    )
    sparse_points = np.zeros((4, 3))
    dense_points = np.zeros((10, 3))

    sparse = estimate_track_geometry(
        sparse_points,
        tracking_result=tracking_result(np.ones(4, dtype=int)),
        track_id=1,
        floor_height_robot_m=0.0,
        config=settings,
    )
    dense = estimate_track_geometry(
        dense_points,
        tracking_result=tracking_result(np.ones(10, dtype=int)),
        track_id=1,
        floor_height_robot_m=0.0,
        config=settings,
    )

    assert sparse.width_m.confidence == pytest.approx(0.3)
    assert dense.width_m.confidence == pytest.approx(0.9)


def test_semantic_association_is_provenance_not_geometry_confidence() -> None:
    points = np.zeros((4, 3))
    association = TrackSemanticAssociation(
        track_id=1,
        status="matched",
        visible_point_count=4,
        detection_index=0,
        mask_hit_count=3,
        overlap_fraction=0.75,
        association_score=0.45,
        projected_median_inside_box=True,
        projected_median_xy=(100.0, 80.0),
    )
    kwargs = {
        "tracking_result": tracking_result(np.ones(4, dtype=int)),
        "track_id": 1,
        "floor_height_robot_m": 0.0,
        "config": exact_config(),
    }

    without_camera = estimate_track_geometry(points, **kwargs)
    with_camera = estimate_track_geometry(
        points,
        association=association,
        **kwargs,
    )

    assert with_camera.width_m.confidence == without_camera.width_m.confidence
    assert any(
        "overlap=0.750" in item
        for item in with_camera.width_m.evidence
    )


def test_mismatched_semantic_association_is_rejected() -> None:
    association = TrackSemanticAssociation(
        track_id=8,
        status="matched",
        visible_point_count=4,
        detection_index=0,
        mask_hit_count=4,
        overlap_fraction=1.0,
        association_score=0.9,
        projected_median_inside_box=True,
        projected_median_xy=(0.0, 0.0),
    )

    with pytest.raises(ValueError, match="must match"):
        estimate_track_geometry(
            np.zeros((4, 3)),
            tracking_result=tracking_result(np.ones(4, dtype=int)),
            track_id=1,
            floor_height_robot_m=0.0,
            association=association,
        )


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((4, 2)),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, np.nan],
            ]
        ),
    ],
)
def test_invalid_point_cloud_is_rejected(points: np.ndarray) -> None:
    with pytest.raises(ValueError):
        estimate_track_geometry(
            points,
            tracking_result=tracking_result(np.ones(4, dtype=int)),
            track_id=1,
            floor_height_robot_m=0.0,
        )


def test_misaligned_tracking_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="must align"):
        estimate_track_geometry(
            np.zeros((4, 3)),
            tracking_result=tracking_result(np.ones(3, dtype=int)),
            track_id=1,
            floor_height_robot_m=0.0,
        )


@pytest.mark.parametrize("track_id", [-1, True, 1.5])
def test_invalid_track_id_is_rejected(track_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        estimate_track_geometry(
            np.zeros((4, 3)),
            tracking_result=tracking_result(np.ones(4, dtype=int)),
            track_id=track_id,  # type: ignore[arg-type]
            floor_height_robot_m=0.0,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"minimum_track_points": 0},
        {
            "minimum_track_points": 5,
            "full_confidence_point_count": 4,
        },
        {"lower_quantile": 0.8, "upper_quantile": 0.2},
        {"range_uncertainty_m": -0.1},
        {"minimum_confidence": 0.8, "maximum_confidence": 0.2},
    ],
)
def test_invalid_geometry_configuration_is_rejected(
    changes: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        exact_config(**changes)
