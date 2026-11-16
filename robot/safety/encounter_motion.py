"""Source-clocked human motion classification; no actor script or CRA evidence changes."""

from dataclasses import asdict, dataclass
import math

import numpy as np


def validated_route_direction(value):
    direction = np.asarray(value, dtype=float)
    if (
        direction.shape != (2,)
        or not np.isfinite(direction).all()
        or not np.isclose(np.linalg.norm(direction), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise ValueError("nominal route direction must be a finite planar unit vector")
    return direction


@dataclass(frozen=True)
class EncounterMotionConfig:
    minimum_points: int = 20
    minimum_observations: int = 3
    maximum_source_gap_seconds: float = 0.35
    stationary_speed_mps: float = 0.12
    moving_speed_mps: float = 0.2
    maximum_speed_mps: float = 2.0
    parallel_enter_degrees: float = 30.0
    parallel_hold_degrees: float = 40.0
    crossing_enter_degrees: float = 60.0
    crossing_hold_degrees: float = 50.0
    maximum_velocity_change_mps: float = 0.35
    maximum_prediction_residual_m: float = 0.2
    maximum_route_turn_degrees: float = 15.0
    minimum_goal_distance_m: float = 0.2

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("minimum_points", "minimum_observations"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 3:
                raise ValueError(f"{name} must be an integer >=3")
        if not (
            self.stationary_speed_mps < self.moving_speed_mps < self.maximum_speed_mps
            and 0
            < self.parallel_enter_degrees
            < self.parallel_hold_degrees
            < self.crossing_hold_degrees
            < self.crossing_enter_degrees
            <= 90
            and self.maximum_route_turn_degrees < 90
        ):
            raise ValueError("invalid speed/angular hysteresis")


class EncounterMotionClassifier:
    """Three consistent current observations; ambiguity immediately disables classification.

    Parallel includes both directions. Angle uses the configured nominal route
    segment when supplied, otherwise the current waypoint bearing, independent
    of chassis heading. State hysteresis is policy only;
    source times and physical measurements are never renewed or borrowed.
    """

    def __init__(self, config=None):
        self.config = config or EncounterMotionConfig()
        if not isinstance(self.config, EncounterMotionConfig):
            raise TypeError("encounter motion config required")
        self.history = {}
        self.last_timestamp = None

    def update(
        self,
        tracking,
        *,
        robot_xy,
        goal_xy,
        human_track_ids,
        nominal_route_direction_world_xy=None,
    ):
        cfg = self.config
        t = float(tracking.timestamp_seconds)
        if (
            not math.isfinite(t)
            or t < 0
            or self.last_timestamp is not None
            and t <= self.last_timestamp
        ):
            raise ValueError("encounter source timestamps must increase")
        robot, goal = (np.asarray(x, dtype=float) for x in (robot_xy, goal_xy))
        if any(x.shape != (2,) or not np.isfinite(x).all() for x in (robot, goal)):
            raise ValueError("finite planar route context required")
        tracks = {x.track_id: x for x in tracking.tracked_obstacles}
        ids = tuple(human_track_ids)
        if (
            len(tracks) != len(tracking.tracked_obstacles)
            or len(set(ids)) != len(ids)
            or any(
                isinstance(k, (bool, np.bool_))
                or not isinstance(k, (int, np.integer))
                or k not in tracks
                for k in ids
            )
        ):
            raise ValueError("unique current verified human identities required")
        delta = goal - robot
        distance = float(np.linalg.norm(delta))
        direction = (
            delta / max(distance, 1e-9)
            if nominal_route_direction_world_xy is None
            else validated_route_direction(nominal_route_direction_world_xy)
        )
        history, rows = {}, {}
        for key in ids:
            track = tracks[key]
            p, v = (
                np.asarray(x, dtype=float)
                for x in (track.centroid_world_xy, track.velocity_world_xy)
            )
            if any(x.shape != (2,) or not np.isfinite(x).all() for x in (p, v)):
                raise ValueError(
                    "finite current observed position and inferred world velocity required"
                )
            speed = float(np.linalg.norm(v))
            old = self.history.get(key)
            reasons = []
            continuous = (
                old is not None
                and 0 < t - old["time"] <= cfg.maximum_source_gap_seconds + 1e-9
            )
            if continuous:
                dt = t - old["time"]
                continuous = (
                    np.linalg.norm(v - old["velocity"])
                    <= cfg.maximum_velocity_change_mps
                    and np.linalg.norm(
                        p - old["position"] - 0.5 * (v + old["velocity"]) * dt
                    )
                    <= cfg.maximum_prediction_residual_m
                    and direction @ old["direction"]
                    >= math.cos(math.radians(cfg.maximum_route_turn_degrees))
                )
            if not track.matched_previous_track or track.observation_count < 3:
                reasons.append("unvalidated_track_continuity")
            if track.point_count < cfg.minimum_points:
                reasons.append("sparse_current_geometry")
            if (
                nominal_route_direction_world_xy is None
                and distance < cfg.minimum_goal_distance_m
            ):
                reasons.append("waypoint_bearing_unresolved_near_goal")
            if speed > cfg.maximum_speed_mps:
                reasons.append("speed_outside_validated_scope")
            angle = None
            raw = "uncertain"
            if not reasons:
                if speed <= cfg.stationary_speed_mps:
                    raw = "stationary"
                elif speed >= cfg.moving_speed_mps:
                    cosine = np.clip(abs(float(v @ direction)) / speed, 0.0, 1.0)
                    angle = math.degrees(math.acos(cosine))
                    prior = old["state"] if continuous else "uncertain"
                    if angle <= (
                        cfg.parallel_hold_degrees
                        if prior == "parallel"
                        else cfg.parallel_enter_degrees
                    ):
                        raw = "parallel"
                    elif angle >= (
                        cfg.crossing_hold_degrees
                        if prior == "crossing"
                        else cfg.crossing_enter_degrees
                    ):
                        raw = "crossing"
                    else:
                        reasons.append("angular_transition_band")
                else:
                    reasons.append("speed_transition_band")
            count = old["count"] + 1 if continuous and old["raw"] == raw else 1
            state = (
                raw
                if raw != "uncertain" and count >= cfg.minimum_observations
                else "uncertain"
            )
            entered = old["entered"] if continuous and old["state"] == state else t
            history[key] = dict(
                time=t,
                position=p.copy(),
                velocity=v.copy(),
                direction=direction.copy(),
                raw=raw,
                count=count,
                state=state,
                entered=entered,
            )
            rows[key] = dict(
                track_id=int(key),
                source_timestamp_seconds=t,
                classification=state,
                candidate_classification=raw,
                consistent_observations=count,
                classification_started_timestamp_seconds=entered,
                acute_route_angle_degrees=angle,
                world_speed_mps=speed,
                velocity_world_xy=v.tolist(),
                parallel_direction=("same" if v @ direction >= 0 else "opposite")
                if state == "parallel"
                else None,
                route_direction_world_xy=direction.tolist(),
                route_direction_basis="active_waypoint_bearing"
                if nominal_route_direction_world_xy is None
                else "configured_obstacle_unaware_nominal_route_segment",
                reasons=reasons,
                status="inferred" if state != "uncertain" else "unavailable",
                basis="current LiDAR tracked world velocity and obstacle-unaware "
                + (
                    "active waypoint bearing"
                    if nominal_route_direction_world_xy is None
                    else "configured nominal route segment"
                ),
            )
        # Missing identities disappear immediately. A reappearing ID must qualify again.
        self.history, self.last_timestamp = history, t
        return dict(
            source_timestamp_seconds=t,
            tracks=rows,
            configuration=asdict(cfg),
            scope="motion-policy classification only; no retained physical evidence, actor truth or collision guarantee",
        )
