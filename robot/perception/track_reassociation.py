"""Conservative geometric reassociation for replacement LiDAR track IDs.

The obstacle tracker intentionally owns geometry and physical motion.  This
module does not create, merge, or relabel tracks.  It only compares retained
geometric snapshots with newly emitted tracks so temporal semantic memory can
decide whether evidence may safely follow an identity change.

Matches are accepted only when they are mutually best and separated from
competing candidates by a configurable cost margin.  Ambiguous cases remain
unmatched and therefore use the downstream conservative ``unknown`` policy.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Iterable

import numpy as np

from robot.safety.obstacle_tracking import TrackedObstacle


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real number while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    """Return one nonnegative integer while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} cannot be negative.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    """Return one positive integer while rejecting booleans."""

    result = _nonnegative_integer(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def _finite_vector2(value: object, *, name: str) -> tuple[float, float]:
    """Return one finite planar vector."""

    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (2,):
        raise ValueError(f"{name} must have shape (2,).")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values.")
    return float(vector[0]), float(vector[1])


@dataclass(frozen=True)
class TrackReassociationConfig:
    """Gates and ambiguity policy for replacement-ID matching."""

    maximum_time_gap_seconds: float = 0.75
    maximum_position_error: float = 0.35
    maximum_velocity_error: float = 1.00
    maximum_point_count_ratio: float = 4.00
    velocity_cost_weight: float = 0.20
    point_count_cost_weight: float = 0.10
    minimum_normalized_cost_margin: float = 0.20

    def __post_init__(self) -> None:
        positive_names = (
            "maximum_time_gap_seconds",
            "maximum_position_error",
            "maximum_velocity_error",
        )
        for name in positive_names:
            value = _finite_real(getattr(self, name), name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")
        ratio = _finite_real(
            self.maximum_point_count_ratio,
            name="maximum_point_count_ratio",
        )
        if ratio <= 1.0:
            raise ValueError(
                "maximum_point_count_ratio must be greater than one."
            )
        for name in (
            "velocity_cost_weight",
            "point_count_cost_weight",
        ):
            value = _finite_real(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative.")
        margin = _finite_real(
            self.minimum_normalized_cost_margin,
            name="minimum_normalized_cost_margin",
        )
        if margin <= 0.0:
            raise ValueError(
                "minimum_normalized_cost_margin must be greater than zero."
            )


@dataclass(frozen=True)
class TrackGeometrySnapshot:
    """World-frame geometry retained for one LiDAR track observation."""

    track_id: int
    timestamp_seconds: float
    centroid_world_xy: tuple[float, float]
    velocity_world_xy: tuple[float, float]
    point_count: int

    def __post_init__(self) -> None:
        track_id = _nonnegative_integer(self.track_id, name="track_id")
        timestamp = _finite_real(
            self.timestamp_seconds,
            name="timestamp_seconds",
        )
        if timestamp < 0.0:
            raise ValueError("timestamp_seconds cannot be negative.")
        centroid = _finite_vector2(
            self.centroid_world_xy,
            name="centroid_world_xy",
        )
        velocity = _finite_vector2(
            self.velocity_world_xy,
            name="velocity_world_xy",
        )
        point_count = _positive_integer(
            self.point_count,
            name="point_count",
        )
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "timestamp_seconds", timestamp)
        object.__setattr__(self, "centroid_world_xy", centroid)
        object.__setattr__(self, "velocity_world_xy", velocity)
        object.__setattr__(self, "point_count", point_count)

    @classmethod
    def from_tracked_obstacle(
        cls,
        track: TrackedObstacle,
        *,
        timestamp_seconds: float,
    ) -> "TrackGeometrySnapshot":
        """Capture validated reassociation geometry from a current track."""

        if not isinstance(track, TrackedObstacle):
            raise TypeError("track must be a TrackedObstacle.")
        return cls(
            track_id=track.track_id,
            timestamp_seconds=timestamp_seconds,
            centroid_world_xy=track.centroid_world_xy,
            velocity_world_xy=track.velocity_world_xy,
            point_count=track.point_count,
        )


@dataclass(frozen=True)
class TrackReassociationMatch:
    """One mutually unambiguous replacement-ID match."""

    previous_track_id: int
    current_track_id: int
    time_gap_seconds: float
    predicted_position_error: float
    velocity_error: float
    point_count_ratio: float
    normalized_cost: float


@dataclass(frozen=True)
class TrackReassociationResult:
    """Accepted matches and ambiguity diagnostics for one update."""

    matches: tuple[TrackReassociationMatch, ...]
    viable_candidate_count: int
    ambiguous_previous_track_ids: tuple[int, ...]
    ambiguous_current_track_ids: tuple[int, ...]

    @property
    def matched_track_count(self) -> int:
        """Return the number of accepted replacement-ID matches."""

        return len(self.matches)


@dataclass(frozen=True)
class _Candidate:
    """One geometrically gated source--target hypothesis."""

    match: TrackReassociationMatch

    @property
    def previous_track_id(self) -> int:
        return self.match.previous_track_id

    @property
    def current_track_id(self) -> int:
        return self.match.current_track_id

    @property
    def cost(self) -> float:
        return self.match.normalized_cost


def reassociate_track_snapshots(
    previous_snapshots: Iterable[TrackGeometrySnapshot],
    current_snapshots: Iterable[TrackGeometrySnapshot],
    *,
    config: TrackReassociationConfig | None = None,
    compatible_id_pairs: Iterable[tuple[int, int]] | None = None,
) -> TrackReassociationResult:
    """Return conservative one-to-one matches between changed track IDs.

    Constant-velocity prediction is used only for association.  A match must
    pass every hard gate, be the lowest-cost option for both its old and new
    IDs, and have sufficient separation from each side's runner-up.
    """

    resolved_config = config or TrackReassociationConfig()
    previous = _validated_unique_snapshots(
        previous_snapshots,
        name="previous_snapshots",
    )
    current = _validated_unique_snapshots(
        current_snapshots,
        name="current_snapshots",
    )
    compatible_pairs = _validated_compatible_pairs(compatible_id_pairs)
    candidates = tuple(
        candidate
        for source in previous
        for target in current
        if compatible_pairs is None
        or (source.track_id, target.track_id) in compatible_pairs
        if (
            candidate := _candidate_for(
                source,
                target,
                config=resolved_config,
            )
        )
        is not None
    )
    by_previous = _group_candidates(
        candidates,
        key="previous",
    )
    by_current = _group_candidates(
        candidates,
        key="current",
    )
    previous_best, ambiguous_previous = _unambiguous_best(
        by_previous,
        minimum_margin=resolved_config.minimum_normalized_cost_margin,
    )
    current_best, ambiguous_current = _unambiguous_best(
        by_current,
        minimum_margin=resolved_config.minimum_normalized_cost_margin,
    )
    matches = tuple(
        candidate.match
        for current_track_id, candidate in sorted(current_best.items())
        if previous_best.get(candidate.previous_track_id) is candidate
        and candidate.current_track_id == current_track_id
    )
    return TrackReassociationResult(
        matches=matches,
        viable_candidate_count=len(candidates),
        ambiguous_previous_track_ids=tuple(sorted(ambiguous_previous)),
        ambiguous_current_track_ids=tuple(sorted(ambiguous_current)),
    )


def _validated_compatible_pairs(
    values: Iterable[tuple[int, int]] | None,
) -> frozenset[tuple[int, int]] | None:
    """Return optional validated ``(previous_id, current_id)`` pairs."""

    if values is None:
        return None
    try:
        pairs = tuple(values)
    except TypeError as error:
        raise TypeError("compatible_id_pairs must be an iterable.") from error
    validated: set[tuple[int, int]] = set()
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(
                "Every compatible_id_pairs item must be a two-item tuple."
            )
        previous_id = _nonnegative_integer(
            pair[0],
            name="compatible previous track ID",
        )
        current_id = _nonnegative_integer(
            pair[1],
            name="compatible current track ID",
        )
        validated.add((previous_id, current_id))
    return frozenset(validated)


def _validated_unique_snapshots(
    values: Iterable[TrackGeometrySnapshot],
    *,
    name: str,
) -> tuple[TrackGeometrySnapshot, ...]:
    """Return validated snapshots with unique IDs."""

    try:
        snapshots = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable.") from error
    seen_ids: set[int] = set()
    for snapshot in snapshots:
        if not isinstance(snapshot, TrackGeometrySnapshot):
            raise TypeError(
                f"Every item in {name} must be a TrackGeometrySnapshot."
            )
        if snapshot.track_id in seen_ids:
            raise ValueError(
                f"{name} contains duplicate track ID {snapshot.track_id}."
            )
        seen_ids.add(snapshot.track_id)
    return snapshots


def _candidate_for(
    source: TrackGeometrySnapshot,
    target: TrackGeometrySnapshot,
    *,
    config: TrackReassociationConfig,
) -> _Candidate | None:
    """Return one gated candidate, or ``None`` when any gate fails."""

    if source.track_id == target.track_id:
        return None
    time_gap = target.timestamp_seconds - source.timestamp_seconds
    if time_gap <= 0.0 or time_gap > config.maximum_time_gap_seconds:
        return None
    source_position = np.asarray(source.centroid_world_xy, dtype=np.float64)
    source_velocity = np.asarray(source.velocity_world_xy, dtype=np.float64)
    target_position = np.asarray(target.centroid_world_xy, dtype=np.float64)
    target_velocity = np.asarray(target.velocity_world_xy, dtype=np.float64)
    predicted_position = source_position + time_gap * source_velocity
    position_error = float(np.linalg.norm(target_position - predicted_position))
    velocity_error = float(np.linalg.norm(target_velocity - source_velocity))
    point_count_ratio = max(source.point_count, target.point_count) / min(
        source.point_count,
        target.point_count,
    )
    if (
        position_error > config.maximum_position_error
        or velocity_error > config.maximum_velocity_error
        or point_count_ratio > config.maximum_point_count_ratio
    ):
        return None
    position_cost = position_error / config.maximum_position_error
    velocity_cost = velocity_error / config.maximum_velocity_error
    point_count_cost = math.log(point_count_ratio) / math.log(
        config.maximum_point_count_ratio
    )
    normalized_cost = (
        position_cost
        + config.velocity_cost_weight * velocity_cost
        + config.point_count_cost_weight * point_count_cost
    )
    return _Candidate(
        TrackReassociationMatch(
            previous_track_id=source.track_id,
            current_track_id=target.track_id,
            time_gap_seconds=time_gap,
            predicted_position_error=position_error,
            velocity_error=velocity_error,
            point_count_ratio=point_count_ratio,
            normalized_cost=normalized_cost,
        )
    )


def _group_candidates(
    candidates: tuple[_Candidate, ...],
    *,
    key: str,
) -> dict[int, list[_Candidate]]:
    """Group candidates by previous or current identifier."""

    grouped: dict[int, list[_Candidate]] = {}
    for candidate in candidates:
        identifier = (
            candidate.previous_track_id
            if key == "previous"
            else candidate.current_track_id
        )
        grouped.setdefault(identifier, []).append(candidate)
    return grouped


def _unambiguous_best(
    grouped: dict[int, list[_Candidate]],
    *,
    minimum_margin: float,
) -> tuple[dict[int, _Candidate], set[int]]:
    """Return best candidates whose runner-up is sufficiently worse."""

    accepted: dict[int, _Candidate] = {}
    ambiguous: set[int] = set()
    for identifier, candidates in grouped.items():
        ranked = sorted(
            candidates,
            key=lambda item: (
                item.cost,
                item.previous_track_id,
                item.current_track_id,
            ),
        )
        if (
            len(ranked) > 1
            and ranked[1].cost - ranked[0].cost < minimum_margin
        ):
            ambiguous.add(identifier)
            continue
        accepted[identifier] = ranked[0]
    return accepted, ambiguous
