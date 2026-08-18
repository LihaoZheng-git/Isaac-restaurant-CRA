"""Temporal semantic memory keyed by persistent LiDAR track identifiers.

Single-frame camera detections can flicker, disappear, or temporarily overlap.
The LiDAR tracker already provides persistent geometry and identity, so this
module accumulates only semantic evidence for each existing ``track_id``.

The memory is deliberately conservative:

* a class needs repeated evidence before it is emitted;
* a short visual dropout may reuse a recent confirmed label;
* competing confirmed classes produce ``unknown`` until one becomes dominant;
* stale evidence expires automatically; and
* camera observations never create a geometric track.
"""

from dataclasses import dataclass, field
import math
from numbers import Integral, Real

from robot.safety.semantic_label_provider import (
    SemanticClassLabel,
    SemanticLabelResult,
    SemanticTrackAssignment,
)
from robot.safety.semantic_obstacle import (
    UNKNOWN_SEMANTIC_CLASS,
    make_unknown_semantic_obstacle,
)


TEMPORAL_LABEL_SOURCE = "temporal_camera_lidar_fusion"


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real number while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    """Return one positive integer while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


@dataclass(frozen=True)
class SemanticTrackMemoryConfig:
    """Temporal evidence, confirmation, and expiration policy."""

    evidence_window_seconds: float = 1.0
    evidence_half_life_seconds: float = 0.50
    minimum_confirming_observations: int = 2
    minimum_observation_confidence: float = 0.05
    minimum_accumulated_evidence: float = 0.12
    minimum_class_evidence_margin: float = 0.05
    track_retention_seconds: float = 2.0
    minimum_time_delta: float = 1e-6

    def __post_init__(self) -> None:
        positive_values = (
            "evidence_window_seconds",
            "evidence_half_life_seconds",
            "track_retention_seconds",
            "minimum_time_delta",
        )
        for name in positive_values:
            value = _finite_real(getattr(self, name), name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")
        if self.track_retention_seconds < self.evidence_window_seconds:
            raise ValueError(
                "track_retention_seconds must be no shorter than "
                "evidence_window_seconds."
            )
        _positive_integer(
            self.minimum_confirming_observations,
            name="minimum_confirming_observations",
        )
        bounded_values = (
            "minimum_observation_confidence",
            "minimum_accumulated_evidence",
            "minimum_class_evidence_margin",
        )
        for name in bounded_values:
            value = _finite_real(getattr(self, name), name=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")


@dataclass(frozen=True)
class TemporalSemanticTrack:
    """Temporal label and evidence diagnostics for one current track."""

    track_id: int
    status: str
    semantic_class: str
    confidence: float
    current_observation_class: str
    current_observation_confidence: float
    supporting_observation_count: int
    evidence_score: float
    evidence_margin: float
    evidence_age_seconds: float | None

    @property
    def is_confirmed(self) -> bool:
        """Return whether this update emits a known semantic class."""

        return self.semantic_class != UNKNOWN_SEMANTIC_CLASS

    @property
    def is_persisted(self) -> bool:
        """Return whether memory bridged a missing or different observation."""

        return self.status == "persisted"


@dataclass(frozen=True)
class SemanticTrackMemoryResult:
    """Temporally filtered labels for one sensor update."""

    timestamp_seconds: float
    semantic_labels: SemanticLabelResult
    tracks: tuple[TemporalSemanticTrack, ...]

    @property
    def confirmed_track_count(self) -> int:
        """Return the number of current tracks with known output labels."""

        return sum(track.is_confirmed for track in self.tracks)

    @property
    def persisted_track_count(self) -> int:
        """Return the number of labels bridged from recent memory."""

        return sum(track.is_persisted for track in self.tracks)


@dataclass(frozen=True)
class _SemanticObservation:
    """One accepted visual-class observation."""

    timestamp_seconds: float
    semantic_class: str
    confidence: float


@dataclass
class _TrackMemoryState:
    """Mutable semantic history retained for one geometric track."""

    track_id: int
    last_seen_timestamp: float
    observations: list[_SemanticObservation] = field(default_factory=list)


@dataclass(frozen=True)
class _ClassEvidence:
    """Age-weighted support for one semantic class."""

    semantic_class: str
    score: float
    observation_count: int
    weighted_confidence: float
    latest_timestamp: float


class SemanticTrackMemory:
    """Accumulate semantic evidence on existing LiDAR ``track_id`` values."""

    def __init__(
        self,
        config: SemanticTrackMemoryConfig | None = None,
    ) -> None:
        self.config = config or SemanticTrackMemoryConfig()
        self._states: dict[int, _TrackMemoryState] = {}
        self._last_update_timestamp: float | None = None

    @property
    def active_track_count(self) -> int:
        """Return the number of retained semantic histories."""

        return len(self._states)

    def reset(self) -> None:
        """Remove all temporal evidence and timestamp history."""

        self._states.clear()
        self._last_update_timestamp = None

    def update(
        self,
        semantic_labels: SemanticLabelResult,
        *,
        timestamp_seconds: float,
    ) -> SemanticTrackMemoryResult:
        """Update semantic evidence for the tracks observed in one frame."""

        if not isinstance(semantic_labels, SemanticLabelResult):
            raise TypeError("semantic_labels must be a SemanticLabelResult.")
        timestamp = self._validate_timestamp(timestamp_seconds)
        self._prune_states(timestamp)
        output_assignments: list[SemanticTrackAssignment] = []
        diagnostics: list[TemporalSemanticTrack] = []
        for assignment in semantic_labels.assignments:
            track = assignment.obstacle.tracked_obstacle
            track_id = track.track_id
            state = self._states.get(track_id)
            if state is None:
                state = _TrackMemoryState(
                    track_id=track_id,
                    last_seen_timestamp=timestamp,
                )
                self._states[track_id] = state
            state.last_seen_timestamp = timestamp
            observed_class = assignment.obstacle.semantic_class
            observed_confidence = assignment.obstacle.semantic_confidence
            if (
                observed_class != UNKNOWN_SEMANTIC_CLASS
                and observed_confidence
                >= self.config.minimum_observation_confidence
            ):
                state.observations.append(
                    _SemanticObservation(
                        timestamp_seconds=timestamp,
                        semantic_class=observed_class,
                        confidence=observed_confidence,
                    )
                )
            self._prune_observations(state, timestamp)
            (
                output_class,
                output_confidence,
                status,
                supporting_count,
                evidence_score,
                evidence_margin,
                evidence_age,
            ) = self._evaluate_state(
                state,
                timestamp=timestamp,
                observed_class=observed_class,
            )
            if output_class == UNKNOWN_SEMANTIC_CLASS:
                output_obstacle = make_unknown_semantic_obstacle(track)
            else:
                output_obstacle = SemanticClassLabel(
                    semantic_class=output_class,
                    confidence=output_confidence,
                    source=TEMPORAL_LABEL_SOURCE,
                ).attach(track)
            output_assignments.append(
                SemanticTrackAssignment(
                    obstacle=output_obstacle,
                    association_distance=assignment.association_distance,
                )
            )
            diagnostics.append(
                TemporalSemanticTrack(
                    track_id=track_id,
                    status=status,
                    semantic_class=output_class,
                    confidence=output_confidence,
                    current_observation_class=observed_class,
                    current_observation_confidence=observed_confidence,
                    supporting_observation_count=supporting_count,
                    evidence_score=evidence_score,
                    evidence_margin=evidence_margin,
                    evidence_age_seconds=evidence_age,
                )
            )
        self._last_update_timestamp = timestamp
        return SemanticTrackMemoryResult(
            timestamp_seconds=timestamp,
            semantic_labels=SemanticLabelResult(
                assignments=tuple(output_assignments),
                unmatched_label_count=semantic_labels.unmatched_label_count,
            ),
            tracks=tuple(diagnostics),
        )

    def _evaluate_state(
        self,
        state: _TrackMemoryState,
        *,
        timestamp: float,
        observed_class: str,
    ) -> tuple[str, float, str, int, float, float, float | None]:
        """Choose one conservative output from the retained evidence."""

        evidence = self._class_evidence(state, timestamp)
        if not evidence:
            return (
                UNKNOWN_SEMANTIC_CLASS,
                0.0,
                "unknown",
                0,
                0.0,
                0.0,
                None,
            )
        ranked = sorted(
            evidence.values(),
            key=lambda item: item.score,
            reverse=True,
        )
        best = ranked[0]
        runner_score = ranked[1].score if len(ranked) > 1 else 0.0
        margin = best.score - runner_score
        eligible = [
            item
            for item in ranked
            if item.observation_count
            >= self.config.minimum_confirming_observations
            and item.score >= self.config.minimum_accumulated_evidence
        ]
        latest_age = timestamp - best.latest_timestamp
        if not eligible:
            return (
                UNKNOWN_SEMANTIC_CLASS,
                0.0,
                "collecting",
                best.observation_count,
                best.score,
                margin,
                latest_age,
            )
        best = eligible[0]
        competing_eligible = eligible[1] if len(eligible) > 1 else None
        eligible_margin = (
            best.score
            - (competing_eligible.score if competing_eligible else 0.0)
        )
        if (
            competing_eligible is not None
            and eligible_margin < self.config.minimum_class_evidence_margin
        ):
            return (
                UNKNOWN_SEMANTIC_CLASS,
                0.0,
                "ambiguous",
                best.observation_count,
                best.score,
                eligible_margin,
                timestamp - best.latest_timestamp,
            )
        total_score = sum(item.score for item in ranked)
        dominance = best.score / total_score if total_score > 0.0 else 0.0
        confidence = min(1.0, best.weighted_confidence * dominance)
        status = (
            "confirmed"
            if observed_class == best.semantic_class
            else "persisted"
        )
        return (
            best.semantic_class,
            confidence,
            status,
            best.observation_count,
            best.score,
            eligible_margin,
            timestamp - best.latest_timestamp,
        )

    def _class_evidence(
        self,
        state: _TrackMemoryState,
        timestamp: float,
    ) -> dict[str, _ClassEvidence]:
        """Return age-weighted evidence grouped by normalized class."""

        observations_by_class: dict[str, list[_SemanticObservation]] = {}
        for observation in state.observations:
            observations_by_class.setdefault(
                observation.semantic_class,
                [],
            ).append(observation)
        evidence: dict[str, _ClassEvidence] = {}
        for semantic_class, observations in observations_by_class.items():
            weights = [
                2.0
                ** (
                    -(
                        timestamp - observation.timestamp_seconds
                    )
                    / self.config.evidence_half_life_seconds
                )
                for observation in observations
            ]
            score = sum(
                weight * observation.confidence
                for weight, observation in zip(
                    weights,
                    observations,
                    strict=True,
                )
            )
            total_weight = sum(weights)
            weighted_confidence = (
                score / total_weight if total_weight > 0.0 else 0.0
            )
            evidence[semantic_class] = _ClassEvidence(
                semantic_class=semantic_class,
                score=score,
                observation_count=len(observations),
                weighted_confidence=weighted_confidence,
                latest_timestamp=max(
                    observation.timestamp_seconds
                    for observation in observations
                ),
            )
        return evidence

    def _prune_states(self, timestamp: float) -> None:
        """Expire old observations and histories for absent tracks."""

        expired_track_ids: list[int] = []
        for track_id, state in self._states.items():
            self._prune_observations(state, timestamp)
            if (
                timestamp - state.last_seen_timestamp
                > self.config.track_retention_seconds
            ):
                expired_track_ids.append(track_id)
        for track_id in expired_track_ids:
            del self._states[track_id]

    def _prune_observations(
        self,
        state: _TrackMemoryState,
        timestamp: float,
    ) -> None:
        """Remove evidence outside the configured temporal window."""

        earliest_timestamp = timestamp - self.config.evidence_window_seconds
        state.observations[:] = [
            observation
            for observation in state.observations
            if observation.timestamp_seconds >= earliest_timestamp
        ]

    def _validate_timestamp(self, value: object) -> float:
        """Return a finite, nonnegative, strictly increasing timestamp."""

        timestamp = _finite_real(value, name="timestamp_seconds")
        if timestamp < 0.0:
            raise ValueError("timestamp_seconds cannot be negative.")
        if self._last_update_timestamp is not None:
            elapsed = timestamp - self._last_update_timestamp
            if elapsed < self.config.minimum_time_delta:
                raise ValueError(
                    "timestamp_seconds must increase by at least "
                    f"{self.config.minimum_time_delta} seconds."
                )
        return timestamp
