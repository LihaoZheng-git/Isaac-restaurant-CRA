"""Opt-in body-extent PRIOR supported by current LiDAR and visual identity.

No actor geometry or state is read. A retained identity is a bounded prior,
never a renewed class observation, mass estimate or motion measurement.
"""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


HUMAN_ENVELOPE_SOURCE = "human_body_envelope_geometry_prior"
HUMAN_ENVELOPE_ID_OFFSET = 3_000_000_000


@dataclass(frozen=True)
class HumanIdentitySelectionPrior:
    """Original visual identity plus a successfully verified current envelope."""

    track_id: int
    original_person_camera_timestamp: float
    original_person_camera_frame_id: int
    selection_camera_timestamp: float
    lidar_timestamp_seconds: float
    current_support_count: int

    def __post_init__(self):
        for value in (self.track_id, self.original_person_camera_frame_id,
                      self.current_support_count):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError("identity prior requires nonnegative integer IDs/count")
        epochs = (self.original_person_camera_timestamp, self.selection_camera_timestamp,
                  self.lidar_timestamp_seconds)
        if any(isinstance(t, bool) or not isinstance(t, Real) or not math.isfinite(t) or t < 0 for t in epochs):
            raise ValueError("identity prior requires finite nonnegative epochs")
        if (not 0 <= self.age_seconds <= .5 + 1e-9
            or not 0 <= self.selection_camera_timestamp - self.lidar_timestamp_seconds <= .25 + 1e-9
            or self.current_support_count < 3):
            raise ValueError("identity selection prior exceeds its bounded source/support contract")

    @property
    def age_seconds(self):
        return self.selection_camera_timestamp - self.original_person_camera_timestamp

    def diagnostic(self):
        from dataclasses import asdict
        return {**asdict(self), "identity_age_at_selection_seconds": self.age_seconds,
                "source": HUMAN_ENVELOPE_SOURCE, "status": "prior",
                "basis": "person_identity_prior_with_verified_current_LiDAR_envelope"}


@dataclass(frozen=True)
class HumanBodyEnvelopeConfig:
    body_radius_m: float = 0.4
    centroid_uncertainty_m: float = 0.2
    maximum_identity_age_seconds: float = 30.0
    maximum_lidar_gap_seconds: float = 0.25
    maximum_centroid_residual_m: float = 0.25
    competitor_distance_m: float = 0.35
    minimum_competitor_point_fraction: float = 0.5
    minimum_points: int = 8

    def __post_init__(self):
        for name, value in vars(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.minimum_points, int) or self.minimum_points < 3:
            raise ValueError("minimum_points must be an integer >= 3")
        if self.minimum_competitor_point_fraction > 1:
            raise ValueError("competitor fraction must be <= 1")


class HumanBodyEnvelope:
    def __init__(self, config, *, recovery_config=None, planning_uncertainty=False):
        if not isinstance(config, HumanBodyEnvelopeConfig):
            raise TypeError("config must be HumanBodyEnvelopeConfig")
        self.config = config
        from robot.perception.human_track_recovery import HumanTrackRecoveryConfig
        if recovery_config is not None and not isinstance(recovery_config, HumanTrackRecoveryConfig):
            raise TypeError("recovery_config must be HumanTrackRecoveryConfig")
        self.recovery_config = recovery_config
        if not isinstance(planning_uncertainty, bool) or planning_uncertainty and recovery_config is None:
            raise ValueError("planning uncertainty requires explicit bounded recovery configuration")
        self.planning_uncertainty = planning_uncertainty
        self.uncertainty_regions = ()
        self.geometry_policies = {}
        self.identities = {}
        self.recovery_audit = []

    def update(self, tracking, fresh_person_cues, *, camera_timestamp_seconds, robot_world_xy=None):
        """Return circles, or block publication rather than erase an old body.

        The caller works on a transaction copy. Legacy unresolved identities
        block publication; the explicit planning-uncertainty mode instead
        retains their reachable geometry without claiming a current identity.
        No source time is renewed by LiDAR-only identity retention.
        """
        now = tracking.timestamp_seconds
        if not math.isfinite(now) or not math.isfinite(camera_timestamp_seconds):
            raise ValueError("human envelope epochs must be finite")
        if (
            not 0
            <= camera_timestamp_seconds - now
            <= self.config.maximum_lidar_gap_seconds
        ):
            raise ValueError("human envelope camera/LiDAR epoch mismatch")
        tracks = {t.track_id: t for t in tracking.tracked_obstacles}
        updated = dict(self.identities)
        result, recovery_audit, uncertain = [], [], []
        def retain_uncertainty(key, previous, reason):
            if not self.planning_uncertainty or previous is None:
                return False
            cfg = self.recovery_config
            if np.linalg.norm(previous['velocity']) > cfg.maximum_human_speed_mps:
                raise ValueError("previous human motion exceeds the uncertainty speed prior")
            from robot.perception.tracking_uncertainty import ReachableTrackRegion
            region = ReachableTrackRegion(key, tuple(previous['center']), previous['lidar_timestamp'],
                previous['camera_timestamp'], previous['camera_frame'], now,
                self.config.body_radius_m+self.config.centroid_uncertainty_m,
                cfg.maximum_human_speed_mps, cfg.checked_horizon_seconds, reason)
            uncertain.append(region)
            recovery_audit.append(dict(decision='planning_geometry_uncertainty', **region.diagnostic()))
            return True
        candidates = {}
        if self.recovery_config is not None:
            # Only missing identities may move to a NEW, uniquely compatible
            # positive track. Existing different identities are never merged.
            for old, previous in self.identities.items():
                dt = now - previous['lidar_timestamp']
                if old in tracks or not 0 < dt <= self.recovery_config.maximum_gap_seconds + 1e-9:
                    continue
                predicted = previous['center'] + dt * previous['velocity']
                candidates[old] = [t.track_id for t in tracks.values()
                    if t.track_id not in self.identities and t.track_id not in fresh_person_cues
                    and t.point_count >= self.config.minimum_points
                    and np.linalg.norm(np.asarray(t.centroid_world_xy) - predicted)
                    <= self.config.maximum_centroid_residual_m]
        for key in sorted(set(self.identities) | set(fresh_person_cues)):
            old_key = key
            previous = self.identities.get(key)
            track = tracks.get(key)
            reassociated = False
            if track is None and len(candidates.get(key, ())) == 1:
                new = candidates[key][0]
                if sum(new in matches for matches in candidates.values()) == 1:
                    key, track, reassociated = new, tracks[new], True
            if track is None:
                if retain_uncertainty(old_key, previous, "missing_positive_track"):
                    continue
                if self.recovery_config is not None:
                    proof = self.recovery_config.missing_path_clearance(previous, now=now,
                        robot_xy=robot_world_xy, body_radius_m=self.config.body_radius_m
                        + self.config.centroid_uncertainty_m)
                    if proof is not None and proof['clearance_m'] > 0:
                        recovery_audit.append(dict(track_id=old_key, decision='temporarily_outside_checked_reach', **proof))
                        continue
                raise ValueError(
                    f"human envelope {key}: current positive LiDAR track missing"
                )
            count = int(np.count_nonzero(tracking.point_track_ids == key))
            center = np.asarray(track.centroid_world_xy, dtype=float)
            velocity = np.asarray(track.velocity_world_xy, dtype=float)
            if (
                count < self.config.minimum_points
                or track.observation_count < 1
                or center.shape != (2,)
                or velocity.shape != (2,)
                or not np.isfinite(center).all()
                or not np.isfinite(velocity).all()
            ):
                if (center.shape == (2,) and velocity.shape == (2,) and np.isfinite(center).all()
                    and np.isfinite(velocity).all()
                    and retain_uncertainty(old_key, previous, "insufficient_current_support")):
                    continue
                raise ValueError(
                    f"human envelope {key}: insufficient current LiDAR support"
                )
            competitors = [
                t.track_id
                for t in tracks.values()
                if t.track_id != key
                and t.point_count
                >= max(
                    self.config.minimum_points,
                    count * self.config.minimum_competitor_point_fraction,
                )
                and np.linalg.norm(np.asarray(t.centroid_world_xy) - center)
                < self.config.competitor_distance_m
            ]
            if competitors:
                if retain_uncertainty(old_key, previous, "ambiguous_current_association"):
                    continue
                raise ValueError(
                    f"human envelope {key}: ambiguous competing tracks {competitors}"
                )
            cue = fresh_person_cues.get(key)
            if cue is not None and previous is not None:
                if cue["camera_frame_id"] < previous["camera_frame"]:
                    raise ValueError("human envelope person frame moved backwards")
                if cue["camera_frame_id"] == previous["camera_frame"]:
                    cue = None
            if cue is not None:
                if cue["camera_timestamp_seconds"] != camera_timestamp_seconds:
                    raise ValueError(
                        "human envelope requires a same-frame accepted person cue"
                    )
                source_time = camera_timestamp_seconds
                source_frame = cue["camera_frame_id"]
                identity_basis = "fresh_accepted_person_SAM_association"
            else:
                dt = now - previous["lidar_timestamp"]
                age = camera_timestamp_seconds - previous["camera_timestamp"]
                residual = np.linalg.norm(
                    center - previous["center"] - dt * previous["velocity"]
                )
                failed_gates = []
                if not track.matched_previous_track and not reassociated:
                    failed_gates.append("unmatched_track")
                if track.observation_count <= previous["observation_count"] and not reassociated:
                    failed_gates.append("nonincreasing_observation_count")
                gap_limit = (self.config.maximum_lidar_gap_seconds if self.recovery_config is None
                             else self.recovery_config.maximum_gap_seconds)
                if not 0 < dt <= gap_limit + 1e-9:
                    failed_gates.append("lidar_source_gap")
                if not 0 <= age <= self.config.maximum_identity_age_seconds:
                    failed_gates.append("original_identity_age")
                if residual > self.config.maximum_centroid_residual_m:
                    failed_gates.append("centroid_prediction_residual")
                if self.recovery_config is not None and (
                    np.linalg.norm(velocity) > self.recovery_config.maximum_human_speed_mps
                    or np.linalg.norm(previous['velocity']) > self.recovery_config.maximum_human_speed_mps):
                    failed_gates.append('recovery_motion_scope')
                if failed_gates:
                    if ('recovery_motion_scope' not in failed_gates and age >= 0 and dt > 0
                        and retain_uncertainty(old_key, previous, ','.join(failed_gates))):
                        continue
                    raise ValueError(
                        f"human envelope {key}: original identity prior expired or continuity unresolved; "
                        f"failed_gates={','.join(failed_gates)}; lidar_source_gap_seconds={dt:.9g}; "
                        f"original_identity_age_seconds={age:.9g}; centroid_prediction_residual_m={residual:.9g}"
                    )
                source_time, source_frame = (
                    previous["camera_timestamp"],
                    previous["camera_frame"],
                )
                identity_basis = (
                    "retained_person_identity_prior_with_current_positive_LiDAR"
                )
                if reassociated or dt > self.config.maximum_lidar_gap_seconds + 1e-9:
                    identity_basis = 'bounded_reacquired_person_identity_prior_with_current_positive_LiDAR'
                    recovery_audit.append(dict(decision='reacquired_current_positive_geometry',
                        previous_track_id=old_key, track_id=key, status='prior',
                        original_person_camera_timestamp=source_time,
                        previous_lidar_timestamp_seconds=previous['lidar_timestamp'],
                        current_lidar_timestamp_seconds=now, source_gap_seconds=dt,
                        centroid_prediction_residual_m=float(residual), changed_track_id=reassociated))
            if reassociated:
                updated.pop(old_key)
            updated[key] = dict(
                camera_timestamp=source_time,
                camera_frame=source_frame,
                lidar_timestamp=now,
                center=center.copy(),
                velocity=velocity.copy(),
                observation_count=track.observation_count,
            )
            result.append(
                dict(
                    track_id=key,
                    envelope_track_id=HUMAN_ENVELOPE_ID_OFFSET + key,
                    center_world_xy=center.tolist(),
                    radius_m=self.config.body_radius_m
                    + self.config.centroid_uncertainty_m,
                    body_radius_prior_m=self.config.body_radius_m,
                    centroid_uncertainty_prior_m=self.config.centroid_uncertainty_m,
                    source=HUMAN_ENVELOPE_SOURCE,
                    status="prior",
                    identity_basis=identity_basis,
                    original_person_camera_timestamp=source_time,
                    original_person_camera_frame_id=source_frame,
                    identity_age_seconds=camera_timestamp_seconds - source_time,
                    lidar_timestamp_seconds=now,
                    current_support_count=count,
                    velocity_world_xy=velocity.tolist(),
                    scope="bounded separated adult; configured extent/centroid bounds, not measured anatomy; no factor-clock extension",
                )
            )
        self.identities = updated
        self.recovery_audit = recovery_audit
        self.uncertainty_regions = tuple(uncertain)
        return result

    def passing_selection_priors(self, tracking, *, camera_timestamp_seconds, maximum_age_seconds):
        """Only a successfully updated current envelope may authorize this ablation.

        This reads retained identity, never supplies physical factors or mutates
        any clock. The longer geometry-identity lease does not extend this cap.
        """
        if (isinstance(maximum_age_seconds, bool) or not math.isfinite(maximum_age_seconds)
            or not 0 < maximum_age_seconds <= .5):
            raise ValueError("selection identity age must be in (0,.5]")
        result = {}
        for track in tracking.tracked_obstacles:
            if any(r.track_id == track.track_id for r in self.uncertainty_regions):
                continue
            entry = self.identities.get(track.track_id)
            if entry is None:
                continue
            if (entry["lidar_timestamp"] != tracking.timestamp_seconds
                or entry["observation_count"] != track.observation_count):
                raise ValueError("selection requires the verified current envelope update")
            age = camera_timestamp_seconds - entry["camera_timestamp"]
            if age < 0:
                raise ValueError("identity selection cannot use future visual evidence")
            if not 0 < age <= maximum_age_seconds + 1e-9:
                continue
            result[track.track_id] = HumanIdentitySelectionPrior(
                track.track_id, entry["camera_timestamp"], entry["camera_frame"],
                camera_timestamp_seconds, tracking.timestamp_seconds,
                int(np.count_nonzero(tracking.point_track_ids == track.track_id)))
        return result


def envelope_grid_points(envelopes, *, grid_config, anchor_xy, anchor_yaw):
    """Rasterize filled prior disks; never remove or relabel observed returns."""
    points, ids = [], []
    c, s = math.cos(anchor_yaw), math.sin(anchor_yaw)
    rotation = np.array([[c, -s], [s, c]])
    step = grid_config.resolution
    origin = np.array([grid_config.minimum_x, grid_config.minimum_y])
    for item in envelopes:
        center = (np.array(item["center_world_xy"]) - anchor_xy) @ rotation
        radius = item["radius_m"]
        low = np.floor((center - radius - origin) / step).astype(int)
        high = np.floor((center + radius - origin) / step).astype(int)
        xy = np.array(
            [
                (x, y)
                for x in range(low[0], high[0] + 1)
                for y in range(low[1], high[1] + 1)
            ]
        )
        cells = origin + (xy + 0.5) * step
        cells = cells[
            np.linalg.norm(cells - center, axis=1) <= radius + step / math.sqrt(2)
        ]
        points.append(cells)
        ids.append(np.full(len(cells), item["envelope_track_id"], dtype=np.int64))
    return (
        (np.concatenate(points), np.concatenate(ids))
        if points
        else (np.empty((0, 2)), np.empty(0, dtype=np.int64))
    )
