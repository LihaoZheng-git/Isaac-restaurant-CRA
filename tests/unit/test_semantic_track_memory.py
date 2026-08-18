import pytest

from robot.perception.semantic_track_memory import (
    TEMPORAL_LABEL_SOURCE,
    SemanticTrackMemory,
    SemanticTrackMemoryConfig,
)
from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.semantic_label_provider import (
    SemanticClassLabel,
    SemanticLabelResult,
    SemanticTrackAssignment,
)
from robot.safety.semantic_obstacle import make_unknown_semantic_obstacle


def make_track(track_id: int = 3) -> TrackedObstacle:
    return TrackedObstacle(
        track_id=track_id,
        centroid_world_xy=(1.0, 2.0),
        centroid_robot_xy=(0.8, 0.1),
        velocity_world_xy=(0.0, 0.0),
        velocity_robot_xy=(0.0, 0.0),
        point_count=12,
        observation_count=4,
        matched_previous_track=True,
    )


def make_result(
    semantic_class: str = "unknown",
    *,
    confidence: float = 0.0,
    track_id: int = 3,
    association_distance: float | None = None,
) -> SemanticLabelResult:
    track = make_track(track_id)
    if semantic_class == "unknown":
        obstacle = make_unknown_semantic_obstacle(track)
    else:
        obstacle = SemanticClassLabel(
            semantic_class,
            confidence=confidence,
            source="camera_lidar_fusion",
        ).attach(track)
    return SemanticLabelResult(
        (
            SemanticTrackAssignment(
                obstacle,
                association_distance=association_distance,
            ),
        )
    )


def test_repeated_observations_confirm_existing_track() -> None:
    memory = SemanticTrackMemory()

    first = memory.update(
        make_result("chair", confidence=0.20),
        timestamp_seconds=0.0,
    )
    second = memory.update(
        make_result("chair", confidence=0.20),
        timestamp_seconds=0.1,
    )

    assert first.semantic_labels.semantic_obstacles[0].is_unknown
    assert first.tracks[0].status == "collecting"
    obstacle = second.semantic_labels.semantic_obstacles[0]
    assert obstacle.track_id == 3
    assert obstacle.semantic_class == "chair"
    assert obstacle.label_source == TEMPORAL_LABEL_SOURCE
    assert obstacle.semantic_confidence == pytest.approx(0.20)
    assert second.tracks[0].status == "confirmed"
    assert second.tracks[0].supporting_observation_count == 2
    assert second.confirmed_track_count == 1


def test_recent_label_persists_through_short_visual_dropout() -> None:
    memory = SemanticTrackMemory()
    memory.update(
        make_result("chair", confidence=0.30),
        timestamp_seconds=0.0,
    )
    memory.update(
        make_result("chair", confidence=0.30),
        timestamp_seconds=0.1,
    )

    dropout = memory.update(make_result(), timestamp_seconds=0.4)

    obstacle = dropout.semantic_labels.semantic_obstacles[0]
    assert obstacle.semantic_class == "chair"
    assert dropout.tracks[0].status == "persisted"
    assert dropout.tracks[0].evidence_age_seconds == pytest.approx(0.3)
    assert dropout.persisted_track_count == 1


def test_stale_label_expires_to_conservative_unknown() -> None:
    memory = SemanticTrackMemory()
    memory.update(
        make_result("chair", confidence=0.30),
        timestamp_seconds=0.0,
    )
    memory.update(
        make_result("chair", confidence=0.30),
        timestamp_seconds=0.1,
    )

    expired = memory.update(make_result(), timestamp_seconds=1.2)

    assert expired.semantic_labels.semantic_obstacles[0].is_unknown
    assert expired.tracks[0].status == "unknown"
    assert expired.tracks[0].evidence_age_seconds is None


def test_one_conflicting_observation_does_not_immediately_flip_class() -> None:
    memory = SemanticTrackMemory()
    memory.update(
        make_result("chair", confidence=0.40),
        timestamp_seconds=0.0,
    )
    memory.update(
        make_result("chair", confidence=0.40),
        timestamp_seconds=0.1,
    )

    conflict = memory.update(
        make_result("person", confidence=0.80),
        timestamp_seconds=0.2,
    )

    assert conflict.semantic_labels.semantic_obstacles[0].semantic_class == "chair"
    assert conflict.tracks[0].status == "persisted"


def test_repeated_stronger_evidence_can_change_class() -> None:
    memory = SemanticTrackMemory()
    memory.update(
        make_result("chair", confidence=0.20),
        timestamp_seconds=0.0,
    )
    memory.update(
        make_result("chair", confidence=0.20),
        timestamp_seconds=0.1,
    )
    memory.update(
        make_result("person", confidence=0.90),
        timestamp_seconds=0.2,
    )

    switched = memory.update(
        make_result("person", confidence=0.90),
        timestamp_seconds=0.3,
    )

    assert switched.semantic_labels.semantic_obstacles[0].semantic_class == "person"
    assert switched.tracks[0].status == "confirmed"


def test_similarly_supported_classes_fall_back_to_unknown() -> None:
    memory = SemanticTrackMemory(
        SemanticTrackMemoryConfig(
            evidence_half_life_seconds=100.0,
            minimum_class_evidence_margin=0.10,
        )
    )
    observations = ("chair", "person", "chair", "person")
    result = None
    for index, semantic_class in enumerate(observations):
        result = memory.update(
            make_result(semantic_class, confidence=0.50),
            timestamp_seconds=0.1 * index,
        )

    assert result is not None
    assert result.semantic_labels.semantic_obstacles[0].is_unknown
    assert result.tracks[0].status == "ambiguous"


def test_low_confidence_observation_is_ignored() -> None:
    memory = SemanticTrackMemory()

    result = memory.update(
        make_result("chair", confidence=0.04),
        timestamp_seconds=0.0,
    )

    assert result.semantic_labels.semantic_obstacles[0].is_unknown
    assert result.tracks[0].status == "unknown"


def test_tracks_are_independent_and_association_distance_is_preserved() -> None:
    memory = SemanticTrackMemory(
        SemanticTrackMemoryConfig(minimum_confirming_observations=1)
    )
    assignments = (
        make_result(
            "chair",
            confidence=0.40,
            track_id=3,
            association_distance=0.02,
        ).assignments[0],
        make_result(
            "person",
            confidence=0.80,
            track_id=8,
            association_distance=0.03,
        ).assignments[0],
    )

    result = memory.update(
        SemanticLabelResult(assignments),
        timestamp_seconds=0.0,
    )

    obstacles = {
        obstacle.track_id: obstacle
        for obstacle in result.semantic_labels.semantic_obstacles
    }
    assert obstacles[3].semantic_class == "chair"
    assert obstacles[8].semantic_class == "person"
    assert [
        assignment.association_distance
        for assignment in result.semantic_labels.assignments
    ] == [0.02, 0.03]


def test_absent_track_history_is_pruned_after_retention_period() -> None:
    memory = SemanticTrackMemory()
    memory.update(
        make_result("chair", confidence=0.30),
        timestamp_seconds=0.0,
    )
    assert memory.active_track_count == 1

    memory.update(SemanticLabelResult(()), timestamp_seconds=2.1)

    assert memory.active_track_count == 0


def test_timestamp_must_increase_and_reset_allows_a_new_sequence() -> None:
    memory = SemanticTrackMemory()
    memory.update(make_result(), timestamp_seconds=1.0)
    with pytest.raises(ValueError, match="must increase"):
        memory.update(make_result(), timestamp_seconds=1.0)

    memory.reset()
    result = memory.update(make_result(), timestamp_seconds=0.0)

    assert result.timestamp_seconds == 0.0
    assert memory.active_track_count == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"evidence_window_seconds": 0.0},
        {"evidence_half_life_seconds": -1.0},
        {"minimum_confirming_observations": 0},
        {"minimum_observation_confidence": 1.1},
        {"minimum_accumulated_evidence": -0.1},
        {"minimum_class_evidence_margin": 1.1},
        {
            "evidence_window_seconds": 2.0,
            "track_retention_seconds": 1.0,
        },
    ],
)
def test_invalid_configuration_is_rejected(overrides: dict[str, float]) -> None:
    with pytest.raises((TypeError, ValueError)):
        SemanticTrackMemoryConfig(**overrides)
