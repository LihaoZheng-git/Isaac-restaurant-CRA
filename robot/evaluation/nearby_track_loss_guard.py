"""Conservative continuity stop for the separated dynamic-actor experiment.

Remembering a positive observation is not observing free space. This watchdog
does not supply geometry, motion, factor evidence or a replacement controller.
Its distances are explicit engineering scope limits, not a safety theorem.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class NearbyTrackLossConfig:
    maximum_missing_seconds: float = 0.15
    stop_distance_m: float = 1.2
    enrollment_distance_m: float = 2.0
    reassociation_distance_m: float = 0.25

    def __post_init__(self):
        for value in vars(self).values():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("track-loss guard limits must be finite and positive")
        if self.enrollment_distance_m < self.stop_distance_m:
            raise ValueError("enroll observations before they enter the stop region")


class NearbyTrackLossGuard:
    def __init__(self, config=None):
        self.config = config if config is not None else NearbyTrackLossConfig()
        if not isinstance(self.config, NearbyTrackLossConfig):
            raise TypeError("expected NearbyTrackLossConfig")
        self.observed = {}
        self.last_source_time = None

    def observe(self, tracks, *, source_time, robot_world_xy):
        if (
            isinstance(source_time, bool)
            or not math.isfinite(source_time)
            or source_time < 0
            or self.last_source_time is not None
            and source_time <= self.last_source_time
        ):
            raise ValueError("track-loss guard source epochs must increase")
        robot = np.asarray(robot_world_xy, dtype=float)
        if robot.shape != (2,) or not np.isfinite(robot).all():
            raise ValueError("robot world position must be a finite planar vector")
        if any(
            isinstance(track.track_id, bool)
            or not isinstance(track.track_id, (int, np.integer))
            or track.track_id < 0
            or track.point_count <= 0
            for track in tracks
        ):
            raise ValueError("the guard requires positive observed geometric tracks")
        current = {
            track.track_id: np.asarray(track.centroid_world_xy, dtype=float)
            for track in tracks
        }
        if len(current) != len(tracks) or any(
            p.shape != (2,) or not np.isfinite(p).all() for p in current.values()
        ):
            raise ValueError(
                "current geometric tracks must have unique IDs and finite centers"
            )
        pending = dict(self.observed)
        missing = {key: value for key, value in pending.items() if key not in current}
        candidates = {
            key: [
                new
                for new, point in current.items()
                if new not in pending
                and np.linalg.norm(point - old[0])
                <= self.config.reassociation_distance_m
            ]
            for key, old in missing.items()
        }
        # Only unambiguous one-to-one positive geometry can resolve an ID change.
        # This never transfers or refreshes old mass/contact/plume/support cues.
        reassociated = set()
        for key, matches in candidates.items():
            if (
                len(matches) == 1
                and sum(matches[0] in others for others in candidates.values()) == 1
            ):
                del pending[key]
                reassociated.add(matches[0])
        for key, point in current.items():
            if (
                key in pending
                or key in reassociated
                or np.linalg.norm(point - robot) <= self.config.enrollment_distance_m
            ):
                pending[key] = (point.copy(), float(source_time))
        blocking = []
        for key, (point, last_seen) in pending.items():
            age = float(source_time - last_seen)
            distance = float(np.linalg.norm(point - robot))
            if (
                age > self.config.maximum_missing_seconds + 1e-9
                and distance <= self.config.stop_distance_m
            ):
                blocking.append(
                    {
                        "track_id": key,
                        "last_observed_world_xy": point.tolist(),
                        "original_last_seen_time": last_seen,
                        "missing_seconds": age,
                        "current_robot_distance_m": distance,
                    }
                )
        self.observed, self.last_source_time = pending, float(source_time)
        return {
            "blocked": bool(blocking),
            "blocking_tracks": blocking,
            "watched_tracks": len(pending),
            "source_timestamp": source_time,
            "scope": "nearby positive LiDAR-track continuity watchdog; lost occupancy is unresolved, not free; bounded separated actors; not a dynamic belief map or general occlusion guarantee",
        }
