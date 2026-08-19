import pytest

from robot.perception.semantic_track_memory import (
    SemanticTrackMemory,
    SemanticTrackMemoryConfig,
)
from robot.perception.track_reassociation import (
    TrackGeometrySnapshot,
    TrackReassociationConfig,
    reassociate_track_snapshots,
)
from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.semantic_label_provider import (
    SemanticClassLabel,
    SemanticLabelResult,
    SemanticTrackAssignment,
)
from robot.safety.semantic_obstacle import make_unknown_semantic_obstacle


def make_snapshot(
    track_id: int,
    *,
    timestamp: float,
    position: tuple[float, float] = (1.0, 2.0),
    velocity: tuple[float, float] = (0.0, 0.0),
    point_count: int = 12,
) -> TrackGeometrySnapshot:
    return TrackGeometrySnapshot(
        track_id=track_id,
        timestamp_seconds=timestamp,
        centroid_world_xy=position,
        velocity_world_xy=velocity,
        point_count=point_count,
    )


def make_track(
    track_id: int,
    *,
    position: tuple[float, float] = (1.0, 2.0),
    velocity: tuple[float, float] = (0.0, 0.0),
    point_count: int = 12,
) -> TrackedObstacle:
    return TrackedObstacle(
        track_id=track_id,
        centroid_world_xy=position,
        centroid_robot_xy=position,
        velocity_world_xy=velocity,
        velocity_robot_xy=velocity,
        point_count=point_count,
        observation_count=1,
        matched_previous_track=False,
    )


def make_assignment(
    track_id: int,
    *,
    semantic_class: str = "unknown",
    confidence: float = 0.0,
    position: tuple[float, float] = (1.0, 2.0),
    velocity: tuple[float, float] = (0.0, 0.0),
    point_count: int = 12,
    association_distance: float | None = None,
) -> SemanticTrackAssignment:
    track = make_track(
        track_id,
        position=position,
        velocity=velocity,
        point_count=point_count,
    )
    obstacle = (
        make_unknown_semantic_obstacle(track)
        if semantic_class == "unknown"
        else SemanticClassLabel(
            semantic_class,
            confidence=confidence,
            source="camera_lidar_fusion",
        ).attach(track)
    )
    return SemanticTrackAssignment(
        obstacle,
        association_distance=association_distance,
    )


def test_motion_prediction_reassociates_one_replacement_id() -> None:
    previous = make_snapshot(
        3,
        timestamp=1.0,
        position=(1.0, 2.0),
        velocity=(0.5, -0.2),
    )
    current = make_snapshot(
        9,
        timestamp=1.2,
        position=(1.10, 1.96),
        velocity=(0.5, -0.2),
    )

    result = reassociate_track_snapshots((previous,), (current,))

    assert result.matched_track_count == 1
    match = result.matches[0]
    assert match.previous_track_id == 3
    assert match.current_track_id == 9
    assert match.time_gap_seconds == pytest.approx(0.2)
    assert match.predicted_position_error == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("previous", "current", "config"),
    [
        (
            make_snapshot(3, timestamp=0.0),
            make_snapshot(9, timestamp=0.8),
            TrackReassociationConfig(maximum_time_gap_seconds=0.5),
        ),
        (
            make_snapshot(3, timestamp=0.0),
            make_snapshot(9, timestamp=0.1, position=(2.0, 2.0)),
            TrackReassociationConfig(maximum_position_error=0.2),
        ),
        (
            make_snapshot(3, timestamp=0.0, velocity=(0.0, 0.0)),
            make_snapshot(
                9,
                timestamp=0.1,
                velocity=(2.0, 0.0),
            ),
            TrackReassociationConfig(maximum_velocity_error=0.5),
        ),
        (
            make_snapshot(3, timestamp=0.0, point_count=5),
            make_snapshot(9, timestamp=0.1, point_count=30),
            TrackReassociationConfig(maximum_point_count_ratio=3.0),
        ),
    ],
)
def test_hard_gates_reject_incompatible_replacement(
    previous: TrackGeometrySnapshot,
    current: TrackGeometrySnapshot,
    config: TrackReassociationConfig,
) -> None:
    result = reassociate_track_snapshots(
        (previous,),
        (current,),
        config=config,
    )

    assert result.matches == ()


def test_two_equally_close_sources_are_conservatively_ambiguous() -> None:
    previous = (
        make_snapshot(3, timestamp=1.0, position=(0.0, -0.05)),
        make_snapshot(4, timestamp=1.0, position=(0.0, 0.05)),
    )
    current = (
        make_snapshot(9, timestamp=1.1, position=(0.0, 0.0)),
    )

    result = reassociate_track_snapshots(previous, current)

    assert result.matches == ()
    assert result.ambiguous_current_track_ids == (9,)


def test_one_source_cannot_be_transferred_to_two_similar_targets() -> None:
    previous = (make_snapshot(3, timestamp=1.0, position=(0.0, 0.0)),)
    current = (
        make_snapshot(8, timestamp=1.1, position=(0.05, 0.0)),
        make_snapshot(9, timestamp=1.1, position=(-0.05, 0.0)),
    )

    result = reassociate_track_snapshots(previous, current)

    assert result.matches == ()
    assert result.ambiguous_previous_track_ids == (3,)


def test_duplicate_snapshot_ids_are_rejected() -> None:
    duplicate = make_snapshot(3, timestamp=1.0)

    with pytest.raises(ValueError, match="duplicate track ID"):
        reassociate_track_snapshots(
            (duplicate, duplicate),
            (make_snapshot(9, timestamp=1.1),),
        )


def test_compatibility_pairs_remove_disallowed_candidates() -> None:
    previous = (
        make_snapshot(3, timestamp=1.0, position=(0.0, -0.05)),
        make_snapshot(4, timestamp=1.0, position=(0.0, 0.05)),
    )
    current = (make_snapshot(9, timestamp=1.1, position=(0.0, 0.0)),)

    result = reassociate_track_snapshots(
        previous,
        current,
        compatible_id_pairs={(3, 9)},
    )

    assert result.matched_track_count == 1
    assert result.matches[0].previous_track_id == 3


@pytest.mark.parametrize(
    "overrides",
    [
        {"maximum_time_gap_seconds": 0.0},
        {"maximum_position_error": -1.0},
        {"maximum_velocity_error": 0.0},
        {"maximum_point_count_ratio": 1.0},
        {"velocity_cost_weight": -0.1},
        {"point_count_cost_weight": -0.1},
        {"minimum_normalized_cost_margin": 0.0},
    ],
)
def test_invalid_reassociation_configuration_is_rejected(
    overrides: dict[str, float],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        TrackReassociationConfig(**overrides)


def test_semantic_memory_moves_confirmed_evidence_to_replacement_id() -> None:
    memory = SemanticTrackMemory(
        SemanticTrackMemoryConfig(minimum_confirming_observations=1)
    )
    memory.update(
        SemanticLabelResult(
            (make_assignment(3, semantic_class="chair", confidence=0.8),)
        ),
        timestamp_seconds=1.0,
    )

    result = memory.update(
        SemanticLabelResult(
            (
                make_assignment(
                    9,
                    position=(1.03, 2.0),
                    association_distance=0.04,
                ),
            )
        ),
        timestamp_seconds=1.1,
    )

    assignment = result.semantic_labels.assignments[0]
    diagnostic = result.tracks[0]
    assert assignment.track_id == 9
    assert assignment.obstacle.semantic_class == "chair"
    assert assignment.association_distance == pytest.approx(0.04)
    assert diagnostic.status == "reassociated"
    assert diagnostic.reassociated_from_track_id == 3
    assert result.reassociated_track_count == 1
    assert result.persisted_track_count == 1
    assert memory.active_track_count == 1


def test_conflicting_current_camera_class_blocks_memory_transfer() -> None:
    memory = SemanticTrackMemory()
    memory.update(
        SemanticLabelResult(
            (make_assignment(3, semantic_class="chair", confidence=0.8),)
        ),
        timestamp_seconds=1.0,
    )
    memory.update(
        SemanticLabelResult(
            (make_assignment(3, semantic_class="chair", confidence=0.8),)
        ),
        timestamp_seconds=1.1,
    )

    result = memory.update(
        SemanticLabelResult(
            (make_assignment(9, semantic_class="person", confidence=0.9),)
        ),
        timestamp_seconds=1.2,
    )

    assert result.semantic_labels.semantic_obstacles[0].is_unknown
    assert result.tracks[0].status == "collecting"
    assert result.tracks[0].reassociated_from_track_id is None
    assert memory.active_track_count == 2


def test_matching_current_camera_class_allows_memory_transfer() -> None:
    memory = SemanticTrackMemory()
    memory.update(
        SemanticLabelResult(
            (make_assignment(3, semantic_class="chair", confidence=0.8),)
        ),
        timestamp_seconds=1.0,
    )
    memory.update(
        SemanticLabelResult(
            (make_assignment(3, semantic_class="chair", confidence=0.8),)
        ),
        timestamp_seconds=1.1,
    )

    result = memory.update(
        SemanticLabelResult(
            (make_assignment(9, semantic_class="chair", confidence=0.8),)
        ),
        timestamp_seconds=1.2,
    )

    assert result.semantic_labels.semantic_obstacles[0].semantic_class == "chair"
    assert result.tracks[0].status == "reassociated"
    assert result.tracks[0].reassociated_from_track_id == 3
    assert memory.active_track_count == 1


def test_ambiguous_semantic_reassociation_falls_back_to_unknown() -> None:
    memory = SemanticTrackMemory(
        SemanticTrackMemoryConfig(minimum_confirming_observations=1)
    )
    memory.update(
        SemanticLabelResult(
            (
                make_assignment(
                    3,
                    semantic_class="chair",
                    confidence=0.8,
                    position=(1.0, 1.95),
                ),
                make_assignment(
                    4,
                    semantic_class="person",
                    confidence=0.8,
                    position=(1.0, 2.05),
                ),
            )
        ),
        timestamp_seconds=1.0,
    )

    result = memory.update(
        SemanticLabelResult((make_assignment(9),)),
        timestamp_seconds=1.1,
    )

    assert result.semantic_labels.semantic_obstacles[0].is_unknown
    assert result.reassociated_track_count == 0


def test_reassociation_can_be_disabled_explicitly() -> None:
    memory = SemanticTrackMemory(
        SemanticTrackMemoryConfig(
            minimum_confirming_observations=1,
            enable_track_reassociation=False,
        )
    )
    memory.update(
        SemanticLabelResult(
            (make_assignment(3, semantic_class="chair", confidence=0.8),)
        ),
        timestamp_seconds=1.0,
    )

    result = memory.update(
        SemanticLabelResult((make_assignment(9),)),
        timestamp_seconds=1.1,
    )

    assert result.semantic_labels.semantic_obstacles[0].is_unknown
    assert result.reassociated_track_count == 0
