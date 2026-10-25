"""Bounded person-relative maneuver hypotheses, never physical evidence.

The frozen curve is built from one qualified observed source. Later fields
verify current identity/motion and keep the original passage deadline. Neither
the short constant-velocity forecast nor kinematic screening is a safety proof.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from robot.control.unicycle_model import wrap_to_pi
from robot.safety.encounter_motion import validated_route_direction


@dataclass(frozen=True)
class PersonRelativeRearConfig:
    maximum_prediction_seconds: float = 3.0
    minimum_prediction_seconds: float = 0.5
    prediction_samples: int = 11
    rear_distance_m: float = 1.2
    preferred_route_offset_m: float = 0.35
    rejoin_distance_m: float = 1.0
    path_lookahead_m: float = 0.25
    maximum_position_residual_m: float = 0.30
    maximum_velocity_change_mps: float = 0.35
    maximum_source_gap_seconds: float = 0.25
    deadline_slack_seconds: float = 0.5
    maximum_linear_velocity: float = 0.2
    maximum_angular_velocity: float = 1.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
        if (
            not isinstance(self.prediction_samples, int)
            or not 2 <= self.prediction_samples <= 25
            or not self.minimum_prediction_seconds
            < self.maximum_prediction_seconds
            <= 3.0
            or self.maximum_source_gap_seconds > 0.25
            or self.deadline_slack_seconds > 0.5
        ):
            raise ValueError(
                "person-relative hypothesis exceeds its bounded timing scope"
            )


def _vector(value, size):
    array = np.asarray(value, float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError("finite person-relative vectors required")
    return array


def _curve(initial, passage, rejoin, route):
    """Two tangent-continuous cubic segments; geometry is only a preference."""
    start = initial[:2]
    heading = np.array([math.cos(initial[2]), math.sin(initial[2])])
    first_length = np.linalg.norm(passage - start) / 3
    last_length = np.linalg.norm(rejoin - passage) / 3
    first = (
        start,
        start + first_length * heading,
        passage - first_length * route,
        passage,
    )
    last = (
        passage,
        passage + last_length * route,
        rejoin - last_length * route,
        rejoin,
    )

    def segment(points):
        t = np.linspace(0.0, 1.0, 17)[:, None]
        a, b, c, d = points
        return (
            (1 - t) ** 3 * a
            + 3 * (1 - t) ** 2 * t * b
            + 3 * (1 - t) * t * t * c
            + t**3 * d
        )

    return np.vstack((segment(first), segment(last)[1:]))


def _screening_seconds(path, initial_yaw, cfg):
    delta = np.diff(path[:17], axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    headings = np.arctan2(delta[:, 1], delta[:, 0])
    turns = np.abs([wrap_to_pi(x) for x in np.diff(np.r_[initial_yaw, headings])])
    return float(
        np.maximum(
            lengths / cfg.maximum_linear_velocity, turns / cfg.maximum_angular_velocity
        ).sum()
    )


@dataclass(frozen=True)
class PersonRelativeRearReference:
    track_id: int
    selection_timestamp: float
    source_timestamp: float
    initial_state: tuple[float, float, float]
    nominal_goal_xy: tuple[float, float]
    route_direction: tuple[float, float]
    original_person_xy: tuple[float, float]
    original_person_velocity: tuple[float, float]
    passage_seconds: float
    config: PersonRelativeRearConfig

    def __post_init__(self):
        if not isinstance(self.config, PersonRelativeRearConfig):
            raise TypeError("person-relative configuration required")
        if (
            isinstance(self.track_id, bool)
            or not isinstance(self.track_id, int)
            or self.track_id < 0
        ):
            raise ValueError("original selected track ID required")
        for name in ("selection_timestamp", "source_timestamp", "passage_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("original finite nonnegative maneuver epochs required")
        if (
            not 0 <= self.source_timestamp - self.selection_timestamp <= 8.0 + 1e-9
            or self.source_timestamp + self.passage_seconds
            > self.selection_timestamp + 8.0 + 1e-9
            or not self.config.minimum_prediction_seconds
            <= self.passage_seconds
            <= self.config.maximum_prediction_seconds
        ):
            raise ValueError("original maneuver/selection timing exceeds its scope")
        for name, size in (
            ("initial_state", 3),
            ("nominal_goal_xy", 2),
            ("route_direction", 2),
            ("original_person_xy", 2),
            ("original_person_velocity", 2),
        ):
            object.__setattr__(self, name, tuple(_vector(getattr(self, name), size)))
        route = validated_route_direction(self.route_direction)
        velocity = np.array(self.original_person_velocity)
        speed = np.linalg.norm(velocity)
        if (
            not 0.2 <= speed <= 2.0
            or abs(route[0] * velocity[1] - route[1] * velocity[0]) < 0.6 * speed
        ):
            raise ValueError("current bounded transverse motion required")
        if (
            np.array(self.original_person_xy) - np.array(self.initial_state[:2])
        ) @ route <= 0:
            raise ValueError("new passage needs a currently ahead person")
        if (self.passage_point - np.array(self.initial_state[:2])) @ route <= 0.1:
            raise ValueError("passage must be ahead of the original robot")
        if (np.array(self.nominal_goal_xy) - self.passage_point) @ route <= 0.1:
            raise ValueError("passage must precede the nominal goal")
        if self.screening_seconds > self.passage_seconds + 1e-9:
            raise ValueError(
                "no passage slot within the configured kinematic screening bound"
            )

    @property
    def predicted_person(self):
        return np.array(self.original_person_xy) + self.passage_seconds * np.array(
            self.original_person_velocity
        )

    @property
    def rear_direction(self):
        velocity = np.array(self.original_person_velocity)
        route = np.array(self.route_direction)
        transverse = velocity - (velocity @ route) * route
        return -transverse / np.linalg.norm(transverse)

    @property
    def passage_point(self):
        route = np.array(self.route_direction)
        goal = np.array(self.nominal_goal_xy)
        base = goal + ((self.predicted_person - goal) @ route) * route
        motion = np.array(self.original_person_velocity)
        motion /= np.linalg.norm(motion)
        # Select a point in the REAR REGION, rather than chase a fixed-distance
        # point moving away faster than the robot. The nominal line is used only
        # to place this separate online reference at the person's crossing plane.
        required = (
            self.config.rear_distance_m + (base - self.predicted_person) @ motion
        ) / (-self.rear_direction @ motion)
        return (
            base
            + max(self.config.preferred_route_offset_m, float(required))
            * self.rear_direction
        )

    @property
    def rejoin_point(self):
        route, goal = np.array(self.route_direction), np.array(self.nominal_goal_xy)
        along = min(
            0.0,
            float((self.passage_point - goal) @ route) + self.config.rejoin_distance_m,
        )
        return goal + along * route

    @property
    def path(self):
        return _curve(
            np.array(self.initial_state),
            self.passage_point,
            self.rejoin_point,
            np.array(self.route_direction),
        )

    @property
    def screening_seconds(self):
        return _screening_seconds(self.path, self.initial_state[2], self.config)

    @property
    def expires_at(self):
        return min(
            self.selection_timestamp + 8.0,
            self.source_timestamp
            + min(
                self.config.maximum_prediction_seconds,
                self.passage_seconds + self.config.deadline_slack_seconds,
            ),
        )

    def validate_current(
        self,
        *,
        track_id,
        selection_timestamp,
        source_timestamp,
        person_xy,
        velocity_xy,
        route_direction,
    ):
        if (
            track_id != self.track_id
            or selection_timestamp != self.selection_timestamp
            or not self.source_timestamp <= source_timestamp <= self.expires_at + 1e-9
            or not np.allclose(route_direction, self.route_direction, rtol=0, atol=1e-9)
        ):
            raise ValueError(
                "person-relative identity/route/original deadline changed or expired"
            )
        dt = source_timestamp - self.source_timestamp
        velocity = _vector(velocity_xy, 2)
        predicted = np.array(self.original_person_xy) + dt * np.array(
            self.original_person_velocity
        )
        if (
            np.linalg.norm(_vector(person_xy, 2) - predicted)
            > self.config.maximum_position_residual_m
            or np.linalg.norm(velocity - np.array(self.original_person_velocity))
            > self.config.maximum_velocity_change_mps
            or velocity @ np.array(self.original_person_velocity) <= 0
        ):
            raise ValueError(
                "current person motion rejects the frozen short-forecast hypothesis"
            )

    def context(self, initial):
        state = _vector(initial, 3)
        path = self.path
        a, delta = path[:-1], np.diff(path, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        fractions = np.clip(
            np.sum((state[:2] - a) * delta, axis=1) / np.maximum(lengths**2, 1e-12),
            0.0,
            1.0,
        )
        nearest = a + fractions[:, None] * delta
        index = int(np.argmin(np.linalg.norm(nearest - state[:2], axis=1)))
        arc = np.r_[0.0, np.cumsum(lengths)]
        progress = arc[index] + fractions[index] * lengths[index]
        target_arc = min(arc[-1], progress + self.config.path_lookahead_m)
        target = np.array([np.interp(target_arc, arc, path[:, k]) for k in range(2)])
        phase = "approach" if progress < arc[16] else "rejoin"
        return dict(
            mode="rear",
            reason="frozen_person_relative_curve",
            target_grid_xy=target.tolist(),
            rear_direction=self.rear_direction.tolist(),
            person_relative_phase=phase,
            person_relative=self.diagnostic(),
            scope="short-forecast maneuver hypothesis, never physical evidence or safe-space geometry",
        )

    def diagnostic(self):
        return dict(
            track_id=self.track_id,
            original_selection_timestamp_seconds=self.selection_timestamp,
            original_source_timestamp_seconds=self.source_timestamp,
            original_person_xy=self.original_person_xy,
            original_person_velocity=self.original_person_velocity,
            predicted_person_xy=self.predicted_person.tolist(),
            passage_point_xy=self.passage_point.tolist(),
            rejoin_point_xy=self.rejoin_point.tolist(),
            path_xy=self.path.tolist(),
            passage_timestamp_seconds=self.source_timestamp + self.passage_seconds,
            expires_at_seconds=self.expires_at,
            kinematic_screening_seconds=self.screening_seconds,
            configuration=asdict(self.config),
            status="inferred",
            basis="frozen_constant_velocity_person_relative_maneuver_hypothesis",
            scope="motion-defined rear, not body facing; no physical-factor/source-clock renewal or safety guarantee",
        )


class PersonRelativeRearMemory:
    """Fresh bounded attempts until one curve freezes; no restart after invalidation."""

    def __init__(self, config):
        if not isinstance(config, PersonRelativeRearConfig):
            raise TypeError("person-relative configuration required")
        self.config, self.key, self.reference, self.last_source = (
            config,
            None,
            None,
            None,
        )
        self.reason = "no_current_selection"
        self.invalidated = False
        self.attempt_count = 0

    def update(
        self,
        *,
        track_id,
        selection_timestamp,
        source_timestamp,
        initial,
        goal,
        person_xy,
        velocity_xy,
        route_direction,
    ):
        if self.last_source is not None and not source_timestamp > self.last_source:
            raise ValueError("person-relative source timestamps must increase")
        gap = None if self.last_source is None else source_timestamp - self.last_source
        self.last_source = source_timestamp
        if track_id is None:
            self.reference, self.reason = None, "no_current_qualified_crossing"
            self.invalidated = True
            return None
        key = (track_id, selection_timestamp)
        if key != self.key:
            self.key, self.reference = key, None
            self.invalidated, self.attempt_count = False, 0
        elif gap is not None and gap > self.config.maximum_source_gap_seconds + 1e-9:
            self.reference, self.reason = None, "source_gap_invalidated_reference"
            self.invalidated = True
        if not 0 <= source_timestamp - selection_timestamp <= 8.0 + 1e-9:
            self.reference, self.reason = None, "original_selection_expired_or_future"
            self.invalidated = True
        if self.reference is None and not self.invalidated:
            self.attempt_count += 1
            self.reason = "no_passage_within_short_kinematic_scope"
            for tau in np.linspace(
                self.config.minimum_prediction_seconds,
                self.config.maximum_prediction_seconds,
                self.config.prediction_samples,
            ):
                try:
                    self.reference = PersonRelativeRearReference(
                        track_id,
                        selection_timestamp,
                        source_timestamp,
                        tuple(initial),
                        tuple(goal),
                        tuple(route_direction),
                        tuple(person_xy),
                        tuple(velocity_xy),
                        float(tau),
                        self.config,
                    )
                except ValueError:
                    continue
                self.reason = "new_frozen_person_relative_reference"
                break
        if self.reference is not None:
            try:
                self.reference.validate_current(
                    track_id=track_id,
                    selection_timestamp=selection_timestamp,
                    source_timestamp=source_timestamp,
                    person_xy=person_xy,
                    velocity_xy=velocity_xy,
                    route_direction=route_direction,
                )
            except ValueError as error:
                self.reference, self.reason = None, str(error)
                self.invalidated = True
        return self.reference

    def diagnostic(self):
        return dict(
            reason=self.reason,
            selection_key=self.key,
            current_source_timestamp_seconds=self.last_source,
            fresh_attempt_count=self.attempt_count,
            invalidated=self.invalidated,
            reference=None if self.reference is None else self.reference.diagnostic(),
            scope="fresh-source search before first frozen curve; original eight-second selection cap; invalidated/expired curves cannot restart",
        )
