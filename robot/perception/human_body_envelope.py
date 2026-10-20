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
    def __init__(self, config):
        if not isinstance(config, HumanBodyEnvelopeConfig):
            raise TypeError("config must be HumanBodyEnvelopeConfig")
        self.config = config
        self.identities = {}

    def update(self, tracking, fresh_person_cues, *, camera_timestamp_seconds):
        """Return circles, or block publication rather than erase an old body.

        The caller works on a transaction copy. Any missing/ambiguous known
        identity prevents publication until a fresh accepted association can
        resolve it. No source time is renewed by LiDAR-only identity retention.
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
        result = []
        for key in sorted(set(self.identities) | set(fresh_person_cues)):
            track = tracks.get(key)
            if track is None:
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
                raise ValueError(
                    f"human envelope {key}: ambiguous competing tracks {competitors}"
                )
            cue = fresh_person_cues.get(key)
            previous = self.identities.get(key)
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
                if (
                    not track.matched_previous_track
                    or track.observation_count <= previous["observation_count"]
                    or not 0 < dt <= self.config.maximum_lidar_gap_seconds + 1e-9
                    or not 0 <= age <= self.config.maximum_identity_age_seconds
                    or residual > self.config.maximum_centroid_residual_m
                ):
                    raise ValueError(
                        f"human envelope {key}: original identity prior expired or continuity unresolved"
                    )
                source_time, source_frame = (
                    previous["camera_timestamp"],
                    previous["camera_frame"],
                )
                identity_basis = (
                    "retained_person_identity_prior_with_current_positive_LiDAR"
                )
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
