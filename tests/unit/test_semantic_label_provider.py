"""Unit tests for replaceable Safe-SAGE semantic-label providers."""

from types import MappingProxyType

import numpy as np
import pytest

from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.semantic_label_provider import (
    NearestWorldPositionSemanticLabelProvider,
    SemanticClassLabel,
    SemanticLabelResult,
    SemanticTrackAssignment,
    TrackIdSemanticLabelProvider,
)


def make_track(
    track_id: int,
    world_xy: tuple[float, float],
) -> TrackedObstacle:
    """Return one deterministic geometric track."""

    return TrackedObstacle(
        track_id=track_id,
        centroid_world_xy=world_xy,
        centroid_robot_xy=world_xy,
        velocity_world_xy=(0.0, 0.6),
        velocity_robot_xy=(0.0, 0.6),
        point_count=8,
        observation_count=3,
        matched_previous_track=True,
    )


def test_semantic_class_label_normalizes_and_attaches() -> None:
    label = SemanticClassLabel(
        " Waiter-With Hot Soup ",
        confidence=0.92,
        source="Manual Test",
    )

    obstacle = label.attach(make_track(4, (2.0, 1.0)))

    assert obstacle.track_id == 4
    assert obstacle.semantic_class == "waiter_with_hot_soup"
    assert obstacle.semantic_confidence == pytest.approx(0.92)
    assert obstacle.label_source == "manual_test"


def test_track_id_provider_selects_only_configured_present_tracks() -> None:
    provider = TrackIdSemanticLabelProvider(
        {
            5: SemanticClassLabel("person"),
            8: SemanticClassLabel("chair"),
        }
    )

    result = provider.assign(
        (make_track(8, (1.0, 0.0)), make_track(3, (0.0, 0.0)))
    )

    assert result.selected_track_ids == (8,)
    assert result.semantic_obstacles[0].semantic_class == "chair"
    assert result.unmatched_label_count == 1
    assert isinstance(provider.labels_by_track_id, MappingProxyType)


def test_nearest_position_provider_uses_world_frame_distance() -> None:
    provider = NearestWorldPositionSemanticLabelProvider(
        np.array([28.2, 5.0]),
        SemanticClassLabel("person"),
        maximum_association_distance=0.5,
    )

    result = provider.assign(
        (
            make_track(2, (27.0, 5.0)),
            make_track(7, (28.1, 5.2)),
        )
    )

    assert result.selected_track_ids == (7,)
    assert result.assignments[0].association_distance == pytest.approx(
        np.hypot(0.1, 0.2)
    )


def test_nearest_position_provider_reports_unmatched_gate() -> None:
    provider = NearestWorldPositionSemanticLabelProvider(
        (10.0, 10.0),
        SemanticClassLabel("person"),
        maximum_association_distance=0.25,
    )

    result = provider.assign((make_track(1, (9.0, 10.0)),))

    assert result.assignments == ()
    assert result.unmatched_label_count == 1


def test_empty_track_collection_is_an_unmatched_label() -> None:
    provider = NearestWorldPositionSemanticLabelProvider(
        (0.0, 0.0),
        SemanticClassLabel("person"),
        maximum_association_distance=0.5,
    )

    result = provider.assign(())

    assert result == SemanticLabelResult((), unmatched_label_count=1)


def test_label_result_rejects_duplicate_track_assignments() -> None:
    obstacle = SemanticClassLabel("person").attach(
        make_track(3, (0.0, 0.0))
    )
    assignment = SemanticTrackAssignment(obstacle)

    with pytest.raises(ValueError, match="duplicate track ID"):
        SemanticLabelResult((assignment, assignment))


def test_target_position_is_immutable() -> None:
    source = np.array([1.0, 2.0])
    provider = NearestWorldPositionSemanticLabelProvider(
        source,
        SemanticClassLabel("person"),
        maximum_association_distance=0.5,
    )
    source[:] = 9.0

    np.testing.assert_allclose(
        provider.target_position_world_xy,
        np.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError):
        provider.target_position_world_xy[0] = 3.0


@pytest.mark.parametrize("confidence", [-0.1, 1.1, np.inf, True])
def test_label_rejects_invalid_confidence(confidence: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SemanticClassLabel("person", confidence=confidence)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "target",
    [(1.0,), (1.0, 2.0, 3.0), (np.nan, 0.0)],
)
def test_nearest_provider_rejects_invalid_target(target: object) -> None:
    with pytest.raises(ValueError):
        NearestWorldPositionSemanticLabelProvider(
            target,
            SemanticClassLabel("person"),
            maximum_association_distance=0.5,
        )


def test_track_collection_rejects_duplicate_ids() -> None:
    provider = TrackIdSemanticLabelProvider(
        {1: SemanticClassLabel("person")}
    )

    with pytest.raises(ValueError, match="duplicate track ID"):
        provider.assign(
            (make_track(1, (0.0, 0.0)), make_track(1, (1.0, 0.0)))
        )
