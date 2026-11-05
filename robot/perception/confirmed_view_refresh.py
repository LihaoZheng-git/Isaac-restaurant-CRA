"""Repeated, score-independent evidence growth for isolated static props.

This is an explicit association assumption, not proof of physical object identity.
An approved update requires fresh RGB/model evidence in the caller. Pending or
rejected candidates never change the saved property, reference, or source clock.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

from robot.perception.camera_lidar_fusion import project_fisheye_polynomial_points
from robot.perception.moving_view_association import compare_moving_view


@dataclass(frozen=True)
class ConfirmedViewRefreshConfig:
    voxel_size_m: float = 0.03
    minimum_new_voxels: int = 8
    minimum_voxel_gain_fraction: float = 0.25
    minimum_span_gain_m: float = 0.06

    def __post_init__(self):
        for value in vars(self).values():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("refresh limits must be finite and positive")
        if not isinstance(self.minimum_new_voxels, int):
            raise ValueError("new voxel count must be an integer")


def growth_evidence(old, candidate, points, view, association, config):
    """Keep every historical anchor; extra sampling density alone is not a gain."""
    reasons = []
    audit = dict(
        rejections=reasons,
        basis="repeated observed surface growth; static isolated-prop assumption",
    )
    reference = old.get("anchor_points", old["points"])
    mask = candidate["proposal"].detection.mask
    quality = candidate["proposal"].model_quality
    old_quality = old.get("model_quality")
    if (
        quality is None
        or old_quality is None
        or not math.isfinite(quality)
        or not math.isfinite(old_quality)
        or quality < old_quality
    ):
        reasons.append("segmentation quality unavailable or lower than saved view")
    if view is None:
        reasons.append("no calibrated current view")
        return audit
    if len(points) < 8 or np.ptp(points, axis=0).max() > association.maximum_extent_m:
        reasons.append("insufficient or oversized current surface")
        return audit
    old_to_new = cKDTree(points).query(reference)[0]
    new_to_old = cKDTree(reference).query(points)[0]
    audit["maximum_original_support_distance_m"] = float(old_to_new.max())
    if np.any(old_to_new > association.support_distance_m):
        reasons.append("not every historical anchor is currently supported")
    projection = project_fisheye_polynomial_points(
        reference,
        camera_from_source=view.camera_from_world,
        calibration=view.calibration,
    )
    if len(projection.source_indices) != len(reference):
        reasons.append("historical anchors leave the current calibrated image")
    elif not mask[
        projection.pixel_indices_xy[:, 1], projection.pixel_indices_xy[:, 0]
    ].all():
        reasons.append(
            "current mask does not contain every projected historical anchor"
        )
    # Count occupied world voxels beyond the old correspondence allowance.
    # Duplicated/denser samples at the same patch cannot manufacture improvement.
    novel = points[new_to_old > association.support_distance_m]
    origin = reference.min(axis=0)
    original_voxels = len(
        np.unique(np.floor((reference - origin) / config.voxel_size_m), axis=0)
    )
    new_voxels = len(
        np.unique(np.floor((novel - origin) / config.voxel_size_m), axis=0)
    )
    span_gain = float(np.max(np.ptp(points, axis=0) - np.ptp(reference, axis=0)))
    audit.update(
        original_voxels=original_voxels,
        new_voxels=new_voxels,
        maximum_span_gain_m=span_gain,
        model_quality=quality,
        original_model_quality=old_quality,
    )
    if (
        new_voxels < config.minimum_new_voxels
        or new_voxels < config.minimum_voxel_gain_fraction * original_voxels
        or span_gain < config.minimum_span_gain_m
    ):
        reasons.append("no meaningful new surface coverage")
    return audit


class ConfirmedViewRefresh:
    def __init__(self, config=None):
        self.config = config or ConfirmedViewRefreshConfig()
        self.pending = {}

    def consider(self, key, old, candidate, points, view, timestamp, association):
        audit = growth_evidence(old, candidate, points, view, association, self.config)
        candidate["diagnostic"]["evidence_refresh"] = audit
        audit.update(
            original_source_time=old["first"],
            current_source_time=timestamp,
            confirmed=False,
        )
        if audit["rejections"]:
            self.pending.pop(key, None)
            return False
        previous = self.pending.get(key)
        matched = False
        if (
            previous is not None
            and 0
            < timestamp - previous["last"]
            <= association.maximum_gap_seconds + 1e-9
        ):
            matched, _, comparison = compare_moving_view(
                previous,
                candidate["proposal"].detection.mask,
                points,
                view,
                association,
            )
            audit["repeat_comparison"] = comparison
        if not matched:
            previous = dict(
                first=timestamp,
                last=timestamp,
                count=1,
                points=points.copy(),
                mask=candidate["proposal"].detection.mask.copy(),
                camera_from_world=view.camera_from_world.copy(),
            )
            self.pending[key] = previous
        else:
            previous["last"] = timestamp
            previous["count"] += 1
        audit.update(observations=previous["count"], proof_start=previous["first"])
        audit["confirmed"] = (
            previous["count"] >= association.minimum_observations
            and timestamp - previous["first"] >= association.minimum_span_seconds - 1e-9
        )
        return audit["confirmed"]

    def prune(self, active_keys, timestamp, maximum_gap):
        self.pending = {
            k: v
            for k, v in self.pending.items()
            if k in active_keys and timestamp - v["last"] <= maximum_gap + 1e-9
        }


def refreshed_reference(old, candidate, points, view, timestamp):
    """Called only after accepted fresh assessment/property storage in the caller."""
    anchors = np.vstack((old.get("anchor_points", old["points"]), points))
    # Exact deduplication only; never discard old anchors via spatial rounding.
    anchors = np.unique(anchors, axis=0)
    return dict(
        first=timestamp,
        proof_start=timestamp,
        last=timestamp,
        count=candidate["diagnostic"]["evidence_refresh"]["observations"],
        mask=candidate["proposal"].detection.mask.copy(),
        points=points.copy(),
        camera_from_world=view.camera_from_world.copy(),
        anchor_points=anchors,
        model_quality=candidate["proposal"].model_quality,
        refresh_history=(
            *old.get("refresh_history", ()),
            dict(
                previous_source_time=old["first"],
                new_source_time=timestamp,
                proof=candidate["diagnostic"]["evidence_refresh"].copy(),
            ),
        ),
    )
