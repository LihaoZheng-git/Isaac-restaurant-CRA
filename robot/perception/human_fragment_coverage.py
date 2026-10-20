"""Opt-in short body-envelope coverage PRIOR; never an observed identity merge.

Store complete positive-return bounds and original risk/time. A configured
point-speed bound expands that region, rather than translating its centroid.
Only a unique currently validated human envelope may cover it. This module
does not supply mass, contact, support, plume, free space or renewed identity.
"""

from dataclasses import dataclass, replace
import math

import numpy as np

from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE


@dataclass(frozen=True)
class HumanFragmentCoverageConfig:
    scope: str
    maximum_missing_seconds: float = .35
    maximum_point_speed_mps: float = 1.
    maximum_observation_gap_seconds: float = .15
    maximum_parent_velocity_change_mps: float = .3
    minimum_observations: int = 4

    def __post_init__(self):
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("fragment coverage requires an explicit unchanged-shape/motion scope")
        for name, value in vars(self).items():
            if name == "scope":
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("coverage limits must be finite positive numbers")
        if (not isinstance(self.minimum_observations, int) or self.minimum_observations < 4
            or self.maximum_missing_seconds > .35
            or self.maximum_observation_gap_seconds > .15):
            raise ValueError("coverage requires >=4 observations and bounded original gaps")


def bounds_corners(low, high):
    return np.array([[low[0], low[1]], [high[0], low[1]],
                     [high[0], high[1]], [low[0], high[1]]], dtype=float)


class HumanFragmentCoverage:
    def __init__(self, config):
        if not isinstance(config, HumanFragmentCoverageConfig):
            raise TypeError("expected HumanFragmentCoverageConfig")
        self.config = config
        self.positive = {}
        self.previous_bodies = {}
        self.last_time = None

    def update(self, tracks, points_world_xy, point_ids, risks, envelopes, *, source_time,
               camera_timestamp_seconds=None):
        camera_time = source_time if camera_timestamp_seconds is None else camera_timestamp_seconds
        points, ids = np.asarray(points_world_xy, float), np.asarray(point_ids)
        current = {t.track_id: t for t in tracks}
        if (not math.isfinite(source_time) or source_time < 0
            or not math.isfinite(camera_time) or not 0 <= camera_time - source_time <= .25 + 1e-9
            or self.last_time is not None and source_time <= self.last_time
            or points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or ids.shape != (len(points),) or not np.issubdtype(ids.dtype, np.integer)
            or len(current) != len(tracks) or np.any(ids < -1)
            or any(int(k) not in current for k in np.unique(ids) if k >= 0)
            or any(np.count_nonzero(ids == k) != t.point_count for k, t in current.items())
            or any(k not in risks or not math.isfinite(risks[k]) or not 0 <= risks[k] <= 1 for k in current)):
            raise ValueError("coverage requires increasing aligned positive geometry and current risks")
        bodies = {}
        for body in envelopes:
            key = body["track_id"]
            center = np.asarray(body["center_world_xy"], float)
            if (key not in current or key in bodies or body["source"] != HUMAN_ENVELOPE_SOURCE
                or body["status"] != "prior" or body["lidar_timestamp_seconds"] != source_time
                or not 0 <= body["original_person_camera_timestamp"] <= camera_time
                or body["current_support_count"] != current[key].point_count
                or body["current_support_count"] < 8 or center.shape != (2,)
                or not np.isfinite(center).all() or not math.isfinite(body["radius_m"])
                or body["radius_m"] <= 0):
                raise ValueError("coverage requires the unique current validated human envelope")
            bodies[key] = body
        pending = {k: v for k, v in self.positive.items()
                   if source_time - v["source_time"] <= self.config.maximum_missing_seconds + 1e-9}
        covered, rejected = [], []
        for key, old in pending.items():
            if key in current:
                continue
            age = source_time - old["source_time"]
            parent = old["parent_id"]
            reason = None
            body, previous_body = bodies.get(parent), self.previous_bodies.get(parent)
            if not old["motion_eligible"] or parent is None:
                reason = "unresolved_original_region_parent_or_motion"
            elif body is None or previous_body is None:
                reason = "no_current_continuous_validated_parent"
            else:
                velocity = np.asarray(current[parent].velocity_world_xy)
                if (not 0 < source_time - previous_body["lidar_timestamp_seconds"]
                    <= self.config.maximum_observation_gap_seconds + 1e-9
                    or np.linalg.norm(velocity) > self.config.maximum_point_speed_mps
                    or np.linalg.norm(velocity - previous_body["velocity_world_xy"])
                    > self.config.maximum_parent_velocity_change_mps):
                    reason = "parent_motion_outside_configured_scope"
            radius = age * self.config.maximum_point_speed_mps
            corners = bounds_corners(old["low"], old["high"])
            if reason is None:
                center = np.asarray(body["center_world_xy"])
                extent = float(np.linalg.norm(corners - center, axis=1).max() + radius)
                if extent > body["radius_m"] + 1e-9:
                    reason = "full_region_plus_motion_not_covered"
                elif any(other != parent and np.linalg.norm(
                    np.maximum(old["low"] - b["center_world_xy"],
                               np.maximum(np.asarray(b["center_world_xy"]) - old["high"], 0.)))
                    <= b["radius_m"] + radius for other, b in bodies.items()):
                    reason = "ambiguous_competing_envelope"
            if reason is not None:
                rejected.append({"track_id": int(key), "original_source_time": old["source_time"], "reason": reason})
                continue
            covered.append({
                "track_id": int(key), "parent_track_id": int(parent),
                "envelope_track_id": body["envelope_track_id"],
                "source": "bounded_human_fragment_coverage_prior", "status": "prior",
                "original_source_time": old["source_time"], "current_source_time": source_time,
                "original_point_count": old["point_count"],
                "original_bounds_min_world_xy": old["low"].tolist(),
                "original_bounds_max_world_xy": old["high"].tolist(),
                "original_risk": old["risk"], "missing_seconds": age,
                "maximum_point_speed_prior_mps": self.config.maximum_point_speed_mps,
                "motion_expansion_m": radius, "maximum_covered_extent_m": extent,
                "current_body_radius_m": body["radius_m"], "scope": self.config.scope,
                "original_person_camera_timestamp": body["original_person_camera_timestamp"],
                "basis": "full_original_positive_region_plus_configured_point_motion; no observed merge or factor transfer",
            })
        # Only actual positive observations replace a source record. Coverage
        # never deletes a missing fragment or extends its original expiry.
        for key, track in current.items():
            owned = points[ids == key]
            low, high = owned.min(axis=0), owned.max(axis=0)
            corners = bounds_corners(low, high)
            parents = [p for p, b in bodies.items() if p != key and
                       np.linalg.norm(corners - b["center_world_xy"], axis=1).max() <= b["radius_m"]]
            previous = self.positive.get(key)
            velocity = np.asarray(track.velocity_world_xy, float)
            if velocity.shape != (2,) or not np.isfinite(velocity).all():
                raise ValueError("finite current velocities required")
            pending[key] = {
                "source_time": float(source_time), "point_count": track.point_count,
                "low": low.copy(), "high": high.copy(), "risk": risks[key],
                "velocity": velocity.copy(), "parent_id": parents[0] if len(parents) == 1 else None,
                "motion_eligible": bool(previous is not None
                    and 0 < source_time - previous["source_time"] <= self.config.maximum_observation_gap_seconds + 1e-9
                    and track.observation_count >= self.config.minimum_observations
                    and np.linalg.norm(velocity) <= self.config.maximum_point_speed_mps
                    and np.linalg.norm(previous["velocity"]) <= self.config.maximum_point_speed_mps),
            }
        self.positive, self.previous_bodies, self.last_time = pending, bodies, float(source_time)
        return {"covered": covered, "rejected": rejected, "scope": self.config.scope}


def apply_coverage_risk_floors(guidance, audit):
    """Raise only applied envelope boundary data, never physical CRA factors."""
    for proof in audit["covered"]:
        key = proof["envelope_track_id"]
        value = guidance[key]
        risk = max(value.risk, proof["original_risk"])
        if value.tangential_strength != 1.25 or not math.isclose(value.boundary_flux, -(.5 + 1.2 * value.risk), abs_tol=1e-12):
            raise ValueError("coverage prior requires unchanged fixed-lambda CRA flux mapping")
        guidance[key] = replace(value, safety_parameters=replace(value.safety_parameters,
            risk=risk, boundary_flux_magnitude=.5 + 1.2 * risk))
    return guidance


def validate_coverage_grid(proof, grid, anchor_xy, anchor_yaw):
    """Check all conservatively intersecting cells, including domain clipping."""
    rotation = np.array([[math.cos(anchor_yaw), -math.sin(anchor_yaw)],
                         [math.sin(anchor_yaw), math.cos(anchor_yaw)]])
    polygon = (bounds_corners(proof["original_bounds_min_world_xy"],
        proof["original_bounds_max_world_xy"]) - anchor_xy) @ rotation
    radius = proof["motion_expansion_m"]
    low, high = polygon.min(axis=0) - radius, polygon.max(axis=0) + radius
    cfg = grid.config
    if (np.any(low < [cfg.minimum_x, cfg.minimum_y])
        or np.any(high >= [cfg.maximum_x, cfg.maximum_y])):
        return {"covered": False, "reason": "coverage_region_outside_field"}
    start = np.floor((low - [cfg.minimum_x, cfg.minimum_y]) / cfg.resolution).astype(int)
    end = np.floor((high - [cfg.minimum_x, cfg.minimum_y]) / cfg.resolution).astype(int)
    cols, rows = np.meshgrid(np.arange(start[0], end[0] + 1), np.arange(start[1], end[1] + 1))
    centers = np.column_stack((cols.ravel(), rows.ravel())) * cfg.resolution
    centers += np.array([cfg.minimum_x, cfg.minimum_y]) + .5 * cfg.resolution
    distances, crosses = [], []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        delta = b - a
        fraction = np.clip((centers - a) @ delta / max(float(delta @ delta), 1e-30), 0, 1)
        distances.append(np.linalg.norm(centers - (a + fraction[:, None] * delta), axis=1))
        crosses.append(delta[0] * (centers[:, 1] - a[1]) - delta[1] * (centers[:, 0] - a[0]))
    crosses = np.array(crosses)
    # Degenerate point/line bounds use edge distances, never infinite interior.
    a, b = polygon[1] - polygon[0], polygon[3] - polygon[0]
    area = abs(float(a[0]*b[1] - a[1]*b[0]))
    inside = ((crosses >= 0).all(axis=0) | (crosses <= 0).all(axis=0)) & (area > 1e-12)
    selected = inside | (np.min(distances, axis=0) <= radius + cfg.resolution / math.sqrt(2) + 1e-9)
    r, c = rows.ravel()[selected], cols.ravel()[selected]
    accepted = bool(len(r) and grid.occupied_mask[r, c].all()
        and (grid.cell_risks[r, c] + 1e-12 >= proof["original_risk"]).all())
    return {"covered": accepted, "checked_cell_count": len(r),
            "reason": None if accepted else "unoccupied_or_lower_risk_coverage_cell"}
