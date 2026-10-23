"""Bounded crossing-side preference from current LiDAR tracking.

This is policy hysteresis, never retained physical evidence, free-space inference
or a collision predictor guarantee. Exact MPC/CBF acceptance remains separate.
"""

from dataclasses import dataclass, asdict
import math

import numpy as np

from robot.safety.safe_sage_guidance_policy import PassingSide


@dataclass(frozen=True)
class MotionAwarePassingConfig:
    require_visual_person_selection: bool = False
    maximum_identity_prior_selection_age_seconds: float = 0.0
    minimum_speed_mps: float = 0.2
    maximum_speed_mps: float = 2.0
    maximum_source_gap_seconds: float = 0.35
    maximum_velocity_change_mps: float = 0.35
    maximum_prediction_residual_m: float = 0.2
    minimum_consistent_observations: int = 3
    minimum_point_count: int = 20
    maximum_distance_m: float = 3.0
    prediction_horizon_seconds: float = 4.0
    maximum_predicted_separation_m: float = 1.6
    minimum_transverse_fraction: float = 0.6
    maximum_encounter_seconds: float = 8.0
    passed_distance_m: float = 0.4

    def __post_init__(self):
        if not isinstance(self.require_visual_person_selection, bool):
            raise TypeError("visual-person selection opt-in must be boolean")
        age = self.maximum_identity_prior_selection_age_seconds
        if (isinstance(age, bool) or not math.isfinite(age) or not 0 <= age <= .5
            or age > 0 and not self.require_visual_person_selection):
            raise ValueError("identity-prior selection requires observed-person mode and age in [0,.5]")
        for name, value in asdict(self).items():
            if name in {"require_visual_person_selection", "maximum_identity_prior_selection_age_seconds"}:
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("minimum_consistent_observations", "minimum_point_count"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 3:
                raise ValueError(f"{name} must be an integer >= 3")
        if (
            self.minimum_speed_mps >= self.maximum_speed_mps
            or self.minimum_transverse_fraction > 1
        ):
            raise ValueError("invalid motion limits")


class MotionAwarePassing:
    def __init__(self, config=None):
        self.config = config or MotionAwarePassingConfig()
        if not isinstance(self.config, MotionAwarePassingConfig):
            raise TypeError("motion passing config required")
        self.history = {}
        self.encounter = None
        self.encounter_person_verified = False
        self.encounter_identity_prior = None
        self.last_timestamp = None

    def update(self, tracking, *, robot_xy, robot_velocity_xy, goal_xy, visual_person_track_ids=None,
               identity_prior_selection_evidence=None, nominal_route_direction_world_xy=None):
        cfg = self.config
        t = float(tracking.timestamp_seconds)
        if (
            not math.isfinite(t)
            or t < 0
            or self.last_timestamp is not None
            and t <= self.last_timestamp
        ):
            raise ValueError("passing source timestamps must increase")
        robot, robot_v, goal = (
            np.asarray(v, dtype=float) for v in (robot_xy, robot_velocity_xy, goal_xy)
        )
        if any(
            v.shape != (2,) or not np.isfinite(v).all() for v in (robot, robot_v, goal)
        ):
            raise ValueError("finite planar passing inputs required")
        delta = goal - robot
        distance = float(np.linalg.norm(delta))
        direction = delta / max(distance, 1e-9)
        if nominal_route_direction_world_xy is not None:
            from robot.safety.encounter_motion import validated_route_direction
            direction = validated_route_direction(nominal_route_direction_world_xy)
        tracks = {track.track_id: track for track in tracking.tracked_obstacles}
        if len(tracks) != len(tracking.tracked_obstacles):
            raise ValueError("ambiguous duplicate track identity")
        if cfg.require_visual_person_selection:
            if visual_person_track_ids is None:
                raise ValueError("explicit current visual-person IDs required")
            eligible = tuple(visual_person_track_ids)
            if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
                   or k not in tracks for k in eligible) or len(set(eligible)) != len(eligible):
                raise ValueError("visual-person IDs must be unique current tracks")
            eligible = set(eligible)
        else:
            if visual_person_track_ids is not None:
                raise ValueError("visual-person IDs require the explicit selection opt-in")
            eligible = set(tracks)
        fresh = set(eligible)
        prior_evidence = {}
        if cfg.maximum_identity_prior_selection_age_seconds > 0:
            from robot.perception.human_body_envelope import HumanIdentitySelectionPrior
            if identity_prior_selection_evidence is None:
                raise ValueError("explicit current envelope selection evidence required")
            for key, evidence in identity_prior_selection_evidence.items():
                if (not isinstance(evidence, HumanIdentitySelectionPrior)
                    or key != evidence.track_id or key not in tracks
                    or evidence.lidar_timestamp_seconds != t
                    or evidence.age_seconds > cfg.maximum_identity_prior_selection_age_seconds + 1e-9
                    or not tracks[key].matched_previous_track
                    or tracks[key].observation_count < 2):
                    raise ValueError("identity selection prior must match current supported track and original age")
                prior_evidence[key] = evidence.diagnostic()
            eligible |= prior_evidence.keys()
        elif identity_prior_selection_evidence is not None:
            raise ValueError("identity selection evidence requires its explicit ablation opt-in")
        candidates, history, rejected = [], {}, {}
        for key, track in tracks.items():
            p, v = (
                np.asarray(track.centroid_world_xy),
                np.asarray(track.velocity_world_xy),
            )
            if (
                p.shape != (2,)
                or v.shape != (2,)
                or not np.isfinite(p).all()
                or not np.isfinite(v).all()
            ):
                raise ValueError("finite current track geometry and velocity required")
            speed = float(np.linalg.norm(v))
            old = self.history.get(key)
            count = 1
            if old is not None:
                dt = t - old[0]
                if (
                    0 < dt <= cfg.maximum_source_gap_seconds + 1e-9
                    and cfg.minimum_speed_mps <= speed <= cfg.maximum_speed_mps
                    and cfg.minimum_speed_mps
                    <= np.linalg.norm(old[2])
                    <= cfg.maximum_speed_mps
                    and np.linalg.norm(v - old[2]) <= cfg.maximum_velocity_change_mps
                    and np.linalg.norm(p - old[1] - 0.5 * (v + old[2]) * dt)
                    <= cfg.maximum_prediction_residual_m
                ):
                    count = old[3] + 1
            history[key] = (t, p.copy(), v.copy(), count)
            rel = p - robot
            rel_v = v - robot_v
            t_closest = -float(rel @ rel_v) / max(float(rel_v @ rel_v), 1e-9)
            separation = float(
                np.linalg.norm(
                    rel + np.clip(t_closest, 0, cfg.prediction_horizon_seconds) * rel_v
                )
            )
            reasons = []
            if (
                not track.matched_previous_track
                or track.observation_count < 3
                or count < cfg.minimum_consistent_observations
            ):
                reasons.append("insufficient_consistent_current_motion")
            if track.point_count < cfg.minimum_point_count:
                reasons.append("sparse_geometry")
            if not cfg.minimum_speed_mps <= speed <= cfg.maximum_speed_mps:
                reasons.append("not_bounded_moving_obstacle")
            if (
                (nominal_route_direction_world_xy is None and distance < 0.2)
                or rel @ direction <= 0
                or np.linalg.norm(rel) > cfg.maximum_distance_m
            ):
                reasons.append("outside_forward_encounter")
            transverse = abs(float(direction[0] * v[1] - direction[1] * v[0])) / max(
                speed, 1e-9
            )
            if transverse < cfg.minimum_transverse_fraction:
                reasons.append("not_transverse_crossing")
            if (
                not 0 < t_closest <= cfg.prediction_horizon_seconds
                or separation > cfg.maximum_predicted_separation_m
            ):
                reasons.append("no_near_term_encounter")
            n = rel / max(float(np.linalg.norm(rel)), 1e-9)
            ccw = np.array([-n[1], n[0]])
            alignment = float(ccw @ v) / max(speed, 1e-9)
            if abs(alignment) < 0.5:
                reasons.append("ambiguous_trailing_tangent")
            if reasons:
                rejected[key] = reasons
            else:
                side = PassingSide.LEFT if alignment < 0 else PassingSide.RIGHT
                candidates.append((key, side, t_closest, separation, alignment))
        decision, preferences = "no_qualified_crossing", {}
        if self.encounter is not None:
            key, side, started, original_velocity = self.encounter
            track = tracks.get(key)
            if any(item[0] != key for item in candidates):
                self.encounter = None
                decision = "released_ambiguous_multiple_crossings"
            elif track is None or t - started > cfg.maximum_encounter_seconds:
                self.encounter = None
                decision = "released_missing_or_expired_target"
            else:
                rel = np.asarray(track.centroid_world_xy) - robot
                v = np.asarray(track.velocity_world_xy)
                if (
                    rel @ direction < -cfg.passed_distance_m
                    or np.linalg.norm(rel) > cfg.maximum_distance_m
                    or np.linalg.norm(v) > cfg.maximum_speed_mps
                    or (
                        np.linalg.norm(v) >= cfg.minimum_speed_mps
                        and v @ original_velocity < 0
                    )
                ):
                    self.encounter = None
                    decision = "released_passed_or_changed_encounter"
                elif (
                    t - self.last_timestamp > cfg.maximum_source_gap_seconds + 1e-9
                    or track.point_count < cfg.minimum_point_count
                    or not track.matched_previous_track
                ):
                    self.encounter = None
                    decision = "released_source_or_identity_gap"
                else:
                    preferences[key] = side
                    decision = "held_current_track_policy_preference"
        if (
            self.encounter is None
            and len(candidates) == 1
            and candidates[0][0] in eligible
            and not decision.startswith("released")
        ):
            key, side, *_ = candidates[0]
            self.encounter = (key, side, t, np.array(tracks[key].velocity_world_xy))
            self.encounter_person_verified = cfg.require_visual_person_selection and key in fresh
            self.encounter_identity_prior = None if key in fresh else prior_evidence.get(key)
            preferences[key] = side
            decision = "selected_trailing_tangent"
        elif self.encounter is None and len(candidates) > 1:
            decision = "ambiguous_multiple_crossings"
        elif self.encounter is None and len(candidates) == 1 and candidates[0][0] not in eligible:
            decision = "unverified_visual_person_target"
        self.history, self.last_timestamp = history, t
        audit = {
            "decision": decision,
            "source_timestamp": t,
            'route_direction_world_xy': direction.tolist(),
            'route_direction_basis': 'active_waypoint_bearing' if nominal_route_direction_world_xy is None else 'configured_obstacle_unaware_nominal_route_segment',
            "encounter_selected_timestamp": None
            if self.encounter is None
            else self.encounter[2],
            "selection_velocity_world_xy": None
            if self.encounter is None
            else self.encounter[3].tolist(),
            "original_selection_visual_person_verified": (
                self.encounter_person_verified if self.encounter is not None else None
            ),
            "current_visual_person_track_ids": sorted(fresh) if cfg.require_visual_person_selection else None,
            "current_identity_prior_selection_evidence": prior_evidence,
            "original_selection_identity_prior": (
                self.encounter_identity_prior if self.encounter is not None else None),
            "preferences": {key: side.value for key, side in preferences.items()},
            "qualified_candidates": [
                {
                    "track_id": key,
                    "side": side.value,
                    "time_to_closest_seconds": tc,
                    "predicted_separation_m": sep,
                    "ccw_motion_alignment": a,
                }
                for key, side, tc, sep, a in candidates
            ],
            "rejections": rejected,
            "configuration": asdict(cfg),
            "basis": "current LiDAR geometry/inferred world velocity; constant-relative-velocity selection hypothesis",
            "scope": "one separated crossing; latched policy preference only; no evidence retention, free-space clearing or guaranteed passing side",
        }
        return preferences, audit
