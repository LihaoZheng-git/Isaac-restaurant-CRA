"""Semantic labels attached to geometrically tracked obstacles.

The LiDAR tracker remains responsible for obstacle identity, position, and
velocity. This module adds a replaceable semantic label source on top of that
geometric track. During controlled PSF-CBF development the source is manual;
later it can be replaced by camera/YOLO fusion without changing downstream
safety-filter interfaces.
"""

from dataclasses import dataclass
import math

from robot.safety.obstacle_tracking import TrackedObstacle


UNKNOWN_SEMANTIC_CLASS = "unknown"
MANUAL_LABEL_SOURCE = "manual"
FALLBACK_LABEL_SOURCE = "fallback"


def normalize_semantic_name(
    value: str,
    *,
    name: str,
) -> str:
    """Return a stable lowercase identifier for a semantic name."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")

    normalized = "_".join(
        value.strip().casefold().replace("-", " ").split()
    )

    if not normalized:
        raise ValueError(f"{name} cannot be empty.")

    return normalized


@dataclass(frozen=True)
class SemanticObstacle:
    """One LiDAR track with an attached semantic classification."""

    tracked_obstacle: TrackedObstacle
    semantic_class: str = UNKNOWN_SEMANTIC_CLASS
    semantic_confidence: float = 0.0
    label_source: str = MANUAL_LABEL_SOURCE

    def __post_init__(self) -> None:
        if not isinstance(
            self.tracked_obstacle,
            TrackedObstacle,
        ):
            raise TypeError(
                "tracked_obstacle must be a TrackedObstacle."
            )

        normalized_class = normalize_semantic_name(
            self.semantic_class,
            name="semantic_class",
        )
        normalized_source = normalize_semantic_name(
            self.label_source,
            name="label_source",
        )

        if isinstance(self.semantic_confidence, bool):
            raise TypeError(
                "semantic_confidence must be a real number."
            )

        try:
            confidence = float(self.semantic_confidence)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "semantic_confidence must be a real number."
            ) from error

        if not math.isfinite(confidence):
            raise ValueError(
                "semantic_confidence must be finite."
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "semantic_confidence must be in [0, 1]."
            )

        object.__setattr__(
            self,
            "semantic_class",
            normalized_class,
        )
        object.__setattr__(
            self,
            "semantic_confidence",
            confidence,
        )
        object.__setattr__(
            self,
            "label_source",
            normalized_source,
        )

    @property
    def track_id(self) -> int:
        """Return the underlying geometric track identifier."""

        return self.tracked_obstacle.track_id

    @property
    def centroid_robot_xy(self) -> tuple[float, float]:
        """Return the obstacle centroid in robot-frame coordinates."""

        return self.tracked_obstacle.centroid_robot_xy

    @property
    def centroid_world_xy(self) -> tuple[float, float]:
        """Return the obstacle centroid in world-frame coordinates."""

        return self.tracked_obstacle.centroid_world_xy

    @property
    def velocity_robot_xy(self) -> tuple[float, float]:
        """Return physical obstacle velocity in robot-frame axes."""

        return self.tracked_obstacle.velocity_robot_xy

    @property
    def velocity_world_xy(self) -> tuple[float, float]:
        """Return physical obstacle velocity in world-frame axes."""

        return self.tracked_obstacle.velocity_world_xy

    @property
    def point_count(self) -> int:
        """Return the number of LiDAR points in the tracked cluster."""

        return self.tracked_obstacle.point_count

    @property
    def is_unknown(self) -> bool:
        """Return whether no recognized semantic class is available."""

        return self.semantic_class == UNKNOWN_SEMANTIC_CLASS


def make_unknown_semantic_obstacle(
    tracked_obstacle: TrackedObstacle,
) -> SemanticObstacle:
    """Wrap a geometric track using the conservative unknown label."""

    return SemanticObstacle(
        tracked_obstacle=tracked_obstacle,
        semantic_class=UNKNOWN_SEMANTIC_CLASS,
        semantic_confidence=0.0,
        label_source=FALLBACK_LABEL_SOURCE,
    )
