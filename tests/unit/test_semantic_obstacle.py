import pytest

from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.semantic_obstacle import (
    SemanticObstacle,
    make_unknown_semantic_obstacle,
    normalize_semantic_name,
)


def make_tracked_obstacle() -> TrackedObstacle:
    """Return one deterministic geometric track for semantic tests."""

    return TrackedObstacle(
        track_id=4,
        centroid_world_xy=(28.2, 5.8),
        centroid_robot_xy=(1.7, 0.8),
        velocity_world_xy=(0.0, -0.6),
        velocity_robot_xy=(0.0, -0.6),
        point_count=9,
        observation_count=12,
        matched_previous_track=True,
    )


def test_semantic_obstacle_preserves_geometric_track() -> None:
    tracked_obstacle = make_tracked_obstacle()

    obstacle = SemanticObstacle(
        tracked_obstacle=tracked_obstacle,
        semantic_class="person",
        semantic_confidence=0.94,
        label_source="manual",
    )

    assert obstacle.tracked_obstacle is tracked_obstacle
    assert obstacle.track_id == 4
    assert obstacle.centroid_robot_xy == (1.7, 0.8)
    assert obstacle.centroid_world_xy == (28.2, 5.8)
    assert obstacle.velocity_robot_xy == (0.0, -0.6)
    assert obstacle.velocity_world_xy == (0.0, -0.6)
    assert obstacle.point_count == 9
    assert obstacle.semantic_class == "person"
    assert obstacle.semantic_confidence == pytest.approx(0.94)
    assert obstacle.label_source == "manual"
    assert not obstacle.is_unknown


def test_semantic_names_are_normalized() -> None:
    obstacle = SemanticObstacle(
        tracked_obstacle=make_tracked_obstacle(),
        semantic_class="  Waiter-With Hot Soup  ",
        semantic_confidence=1,
        label_source="  Manual Label  ",
    )

    assert obstacle.semantic_class == "waiter_with_hot_soup"
    assert obstacle.semantic_confidence == pytest.approx(1.0)
    assert obstacle.label_source == "manual_label"


def test_unknown_wrapper_uses_unclassified_defaults() -> None:
    obstacle = make_unknown_semantic_obstacle(
        make_tracked_obstacle()
    )

    assert obstacle.semantic_class == "unknown"
    assert obstacle.semantic_confidence == pytest.approx(0.0)
    assert obstacle.label_source == "fallback"
    assert obstacle.is_unknown


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Dining Table", "dining_table"),
        ("HOT-SOUP", "hot_soup"),
        ("person", "person"),
    ],
)
def test_normalize_semantic_name(
    value: str,
    expected: str,
) -> None:
    assert normalize_semantic_name(
        value,
        name="test_name",
    ) == expected


@pytest.mark.parametrize(
    "confidence",
    [-0.01, 1.01, float("inf"), float("nan")],
)
def test_invalid_confidence_is_rejected(
    confidence: float,
) -> None:
    with pytest.raises(ValueError):
        SemanticObstacle(
            tracked_obstacle=make_tracked_obstacle(),
            semantic_class="person",
            semantic_confidence=confidence,
        )


@pytest.mark.parametrize(
    "confidence",
    [True, "high", None],
)
def test_non_numeric_confidence_is_rejected(
    confidence: object,
) -> None:
    with pytest.raises(TypeError, match="real number"):
        SemanticObstacle(
            tracked_obstacle=make_tracked_obstacle(),
            semantic_class="person",
            semantic_confidence=confidence,
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("semantic_class", "   "),
        ("label_source", ""),
    ],
)
def test_empty_semantic_names_are_rejected(
    field_name: str,
    field_value: str,
) -> None:
    values = {
        "tracked_obstacle": make_tracked_obstacle(),
        "semantic_class": "person",
        "semantic_confidence": 1.0,
        "label_source": "manual",
    }
    values[field_name] = field_value

    with pytest.raises(ValueError, match="cannot be empty"):
        SemanticObstacle(**values)


def test_non_string_semantic_name_is_rejected() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        SemanticObstacle(
            tracked_obstacle=make_tracked_obstacle(),
            semantic_class=42,
            semantic_confidence=1.0,
        )


def test_non_track_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="TrackedObstacle"):
        SemanticObstacle(
            tracked_obstacle="track 4",
            semantic_class="person",
            semantic_confidence=1.0,
        )
