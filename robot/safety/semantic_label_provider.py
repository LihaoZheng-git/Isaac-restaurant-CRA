"""Replaceable semantic-label providers for geometric obstacle tracks.

LiDAR clustering and tracking establish obstacle identity and motion.  A
semantic-label provider only attaches class information to those existing
tracks; it never creates geometry.  Controlled demos can therefore use a
manual position association today, while a later camera/YOLO fusion module
can implement the same protocol without changing field construction or the
safety controller.
"""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np

from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.semantic_obstacle import (
    MANUAL_LABEL_SOURCE,
    SemanticObstacle,
    normalize_semantic_name,
)


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real number while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _world_point_xy(value: object, *, name: str) -> np.ndarray:
    """Return one immutable finite world-frame XY point."""

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,):
        raise ValueError(f"{name} must have shape (2,).")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    result = np.array(result, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _tracked_obstacles(
    value: Iterable[TrackedObstacle],
) -> tuple[TrackedObstacle, ...]:
    """Materialize and validate one track collection."""

    try:
        tracks = tuple(value)
    except TypeError as error:
        raise TypeError("tracked_obstacles must be an iterable.") from error
    seen_ids: set[int] = set()
    for track in tracks:
        if not isinstance(track, TrackedObstacle):
            raise TypeError(
                "Every tracked obstacle must be a TrackedObstacle."
            )
        if track.track_id in seen_ids:
            raise ValueError(
                f"tracked_obstacles contains duplicate track ID {track.track_id}."
            )
        seen_ids.add(track.track_id)
    return tracks


@dataclass(frozen=True)
class SemanticClassLabel:
    """One class prediction independent of a geometric track."""

    semantic_class: str
    confidence: float = 1.0
    source: str = MANUAL_LABEL_SOURCE

    def __post_init__(self) -> None:
        semantic_class = normalize_semantic_name(
            self.semantic_class,
            name="semantic_class",
        )
        source = normalize_semantic_name(self.source, name="source")
        confidence = _finite_real(self.confidence, name="confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1].")
        object.__setattr__(self, "semantic_class", semantic_class)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source", source)

    def attach(self, track: TrackedObstacle) -> SemanticObstacle:
        """Attach this classification to one existing geometric track."""

        return SemanticObstacle(
            tracked_obstacle=track,
            semantic_class=self.semantic_class,
            semantic_confidence=self.confidence,
            label_source=self.source,
        )


@dataclass(frozen=True)
class SemanticTrackAssignment:
    """One semantic track plus optional association diagnostics."""

    obstacle: SemanticObstacle
    association_distance: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.obstacle, SemanticObstacle):
            raise TypeError("obstacle must be a SemanticObstacle.")
        if self.association_distance is None:
            return
        distance = _finite_real(
            self.association_distance,
            name="association_distance",
        )
        if distance < 0.0:
            raise ValueError("association_distance cannot be negative.")
        object.__setattr__(self, "association_distance", distance)

    @property
    def track_id(self) -> int:
        """Return the associated geometric track ID."""

        return self.obstacle.track_id


@dataclass(frozen=True)
class SemanticLabelResult:
    """Validated semantic assignments produced for one sensor frame."""

    assignments: tuple[SemanticTrackAssignment, ...]
    unmatched_label_count: int = 0

    def __post_init__(self) -> None:
        try:
            assignments = tuple(self.assignments)
        except TypeError as error:
            raise TypeError("assignments must be an iterable.") from error
        seen_track_ids: set[int] = set()
        for assignment in assignments:
            if not isinstance(assignment, SemanticTrackAssignment):
                raise TypeError(
                    "Every assignment must be a SemanticTrackAssignment."
                )
            if assignment.track_id in seen_track_ids:
                raise ValueError(
                    "assignments contains duplicate track ID "
                    f"{assignment.track_id}."
                )
            seen_track_ids.add(assignment.track_id)
        unmatched_count = self.unmatched_label_count
        if isinstance(unmatched_count, bool) or not isinstance(
            unmatched_count,
            int,
        ):
            raise TypeError("unmatched_label_count must be an integer.")
        if unmatched_count < 0:
            raise ValueError("unmatched_label_count cannot be negative.")
        object.__setattr__(self, "assignments", assignments)

    @property
    def semantic_obstacles(self) -> tuple[SemanticObstacle, ...]:
        """Return semantic obstacles in assignment order."""

        return tuple(assignment.obstacle for assignment in self.assignments)

    @property
    def selected_track_ids(self) -> tuple[int, ...]:
        """Return track IDs selected for semantic field construction."""

        return tuple(assignment.track_id for assignment in self.assignments)


@runtime_checkable
class SemanticLabelProvider(Protocol):
    """Interface implemented by manual labels and future vision fusion."""

    def assign(
        self,
        tracked_obstacles: Iterable[TrackedObstacle],
    ) -> SemanticLabelResult:
        """Attach labels to zero or more existing geometric tracks."""


class TrackIdSemanticLabelProvider:
    """Attach configured labels to matching tracker IDs."""

    def __init__(self, labels_by_track_id: Mapping[int, SemanticClassLabel]) -> None:
        if not isinstance(labels_by_track_id, Mapping):
            raise TypeError("labels_by_track_id must be a mapping.")
        labels: dict[int, SemanticClassLabel] = {}
        for track_id, label in labels_by_track_id.items():
            if isinstance(track_id, bool) or not isinstance(track_id, int):
                raise TypeError("Semantic-label track IDs must be integers.")
            if track_id < 0:
                raise ValueError("Semantic-label track IDs cannot be negative.")
            if not isinstance(label, SemanticClassLabel):
                raise TypeError(
                    "Every configured label must be a SemanticClassLabel."
                )
            labels[track_id] = label
        self._labels = MappingProxyType(labels)

    @property
    def labels_by_track_id(self) -> Mapping[int, SemanticClassLabel]:
        """Return the immutable configured mapping."""

        return self._labels

    def assign(
        self,
        tracked_obstacles: Iterable[TrackedObstacle],
    ) -> SemanticLabelResult:
        """Attach labels to the configured IDs present in this frame."""

        tracks = _tracked_obstacles(tracked_obstacles)
        tracks_by_id = {track.track_id: track for track in tracks}
        assignments = tuple(
            SemanticTrackAssignment(self._labels[track_id].attach(tracks_by_id[track_id]))
            for track_id in sorted(self._labels)
            if track_id in tracks_by_id
        )
        return SemanticLabelResult(
            assignments=assignments,
            unmatched_label_count=len(self._labels) - len(assignments),
        )


class NearestWorldPositionSemanticLabelProvider:
    """Controlled-test provider that labels the track nearest a known point.

    The known point is used only to decide which real LiDAR track receives a
    manual label.  It is not returned as geometry and is never supplied to the
    field solver or safety controller.
    """

    def __init__(
        self,
        target_position_world_xy: object,
        label: SemanticClassLabel,
        *,
        maximum_association_distance: float,
    ) -> None:
        self._target_position_world_xy = _world_point_xy(
            target_position_world_xy,
            name="target_position_world_xy",
        )
        if not isinstance(label, SemanticClassLabel):
            raise TypeError("label must be a SemanticClassLabel.")
        maximum_distance = _finite_real(
            maximum_association_distance,
            name="maximum_association_distance",
        )
        if maximum_distance <= 0.0:
            raise ValueError(
                "maximum_association_distance must be greater than zero."
            )
        self.label = label
        self.maximum_association_distance = maximum_distance

    @property
    def target_position_world_xy(self) -> np.ndarray:
        """Return the immutable controlled association point."""

        return self._target_position_world_xy

    def assign(
        self,
        tracked_obstacles: Iterable[TrackedObstacle],
    ) -> SemanticLabelResult:
        """Label the nearest track when it lies inside the association gate."""

        tracks = _tracked_obstacles(tracked_obstacles)
        if not tracks:
            return SemanticLabelResult((), unmatched_label_count=1)
        distances = np.asarray(
            [
                np.linalg.norm(
                    np.asarray(track.centroid_world_xy, dtype=np.float64)
                    - self._target_position_world_xy
                )
                for track in tracks
            ],
            dtype=np.float64,
        )
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        if best_distance > self.maximum_association_distance:
            return SemanticLabelResult((), unmatched_label_count=1)
        return SemanticLabelResult(
            (
                SemanticTrackAssignment(
                    obstacle=self.label.attach(tracks[best_index]),
                    association_distance=best_distance,
                ),
            )
        )
