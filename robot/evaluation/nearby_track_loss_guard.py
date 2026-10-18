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
    allow_observed_fragment_merges: bool = False
    positive_merge_distance_m: float = 0.10
    minimum_positive_merge_returns: int = 3

    def __post_init__(self):
        if not isinstance(self.allow_observed_fragment_merges, bool):
            raise TypeError("fragment merge opt-in must be boolean")
        for name, value in vars(self).items():
            if name == "allow_observed_fragment_merges":
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("track-loss guard limits must be finite and positive")
        if self.enrollment_distance_m < self.stop_distance_m:
            raise ValueError("enroll observations before they enter the stop region")
        if (
            self.positive_merge_distance_m > self.reassociation_distance_m
            or isinstance(self.minimum_positive_merge_returns, bool)
            or not isinstance(self.minimum_positive_merge_returns, int)
        ):
            raise ValueError("invalid positive merge geometry gates")


class NearbyTrackLossGuard:
    def __init__(self, config=None):
        self.config = config if config is not None else NearbyTrackLossConfig()
        if not isinstance(self.config, NearbyTrackLossConfig):
            raise TypeError("expected NearbyTrackLossConfig")
        self.observed = {}
        self.last_source_time = None

    def observe(
        self,
        tracks,
        *,
        source_time,
        robot_world_xy,
        current_points_world_xy=None,
        current_point_track_ids=None,
        current_points_source_time=None,
    ):
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
        geometry = None
        if any(
            v is not None
            for v in (
                current_points_world_xy,
                current_point_track_ids,
                current_points_source_time,
            )
        ):
            xyz = np.asarray(current_points_world_xy, dtype=float)
            ids = np.asarray(current_point_track_ids)
            if (
                xyz.ndim != 2
                or xyz.shape[1] != 2
                or not np.isfinite(xyz).all()
                or ids.shape != (len(xyz),)
                or not np.issubdtype(ids.dtype, np.integer)
                or current_points_source_time is None
                or not math.isfinite(current_points_source_time)
                or abs(current_points_source_time - source_time) > 1e-9
                or np.any(ids < -1)
                or any(int(k) not in current for k in np.unique(ids) if k >= 0)
                or any(
                    np.count_nonzero(ids == t.track_id) != t.point_count for t in tracks
                )
            ):
                raise ValueError(
                    "fragment merge requires aligned current positive sensor geometry"
                )
            geometry = (xyz, ids)
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
                if self.config.allow_observed_fragment_merges and geometry is not None:
                    xyz, ids = geometry
                    near = ids[
                        np.linalg.norm(xyz - missing[key][0], axis=1)
                        <= self.config.positive_merge_distance_m
                    ]
                    supported = [
                        int(n)
                        for n in np.unique(near)
                        if np.count_nonzero(near == n)
                        >= self.config.minimum_positive_merge_returns
                    ]
                    if supported != matches:
                        continue
                del pending[key]
                reassociated.add(matches[0])
        # An articulated body can split then rejoin a still-observed cluster.
        # Resolve this only with a unique current cluster and multiple actually
        # observed returns at the lost fragment's old location. No free-space,
        # old body box, factor, semantic identity or actor truth is inferred.
        merged = []
        if self.config.allow_observed_fragment_merges and geometry is not None:
            xyz, ids = geometry
            proposals = {}
            for key, (old_point, old_time) in missing.items():
                if key not in pending:
                    continue
                nearby_ids = ids[
                    np.linalg.norm(xyz - old_point, axis=1)
                    <= self.config.positive_merge_distance_m
                ]
                positive_ids = [
                    int(n)
                    for n in np.unique(nearby_ids)
                    if np.count_nonzero(nearby_ids == n)
                    >= self.config.minimum_positive_merge_returns
                ]
                choices = [
                    new
                    for new, point in current.items()
                    if new in pending
                    and new != key
                    and abs(pending[new][1] - old_time)
                    <= self.config.maximum_missing_seconds + 1e-9
                    and np.linalg.norm(point - old_point)
                    <= self.config.reassociation_distance_m
                    and positive_ids == [new]
                ]
                proposals[key] = choices
            for key, choices in proposals.items():
                if (
                    len(choices) == 1
                    and sum(choices[0] in v for v in proposals.values()) == 1
                ):
                    new = choices[0]
                    old_point, old_time = pending.pop(key)
                    merged.append(
                        {
                            "lost_track_id": int(key),
                            "current_track_id": int(new),
                            "original_last_seen_time": old_time,
                            "current_source_time": source_time,
                            "positive_returns_near_old_centroid": int(
                                np.count_nonzero(
                                    (ids == new)
                                    & (
                                        np.linalg.norm(xyz - old_point, axis=1)
                                        <= self.config.positive_merge_distance_m
                                    )
                                )
                            ),
                        }
                    )
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
            "observed_fragment_merges": merged,
            "scope": "nearby positive LiDAR-track continuity watchdog; lost occupancy is unresolved, not free; bounded separated actors; not a dynamic belief map or general occlusion guarantee",
        }
