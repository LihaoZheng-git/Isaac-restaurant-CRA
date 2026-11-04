"""Opt-in static-prop correspondence under bounded ego/viewpoint changes.

An affine mask warp inferred from original positive 3D support is an association
PRIOR, not a dense depth reconstruction. Missing old samples remain unobserved;
this module never creates or deletes occupancy and never estimates hidden size.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy.spatial import cKDTree

from robot.perception.camera_lidar_fusion import project_fisheye_polynomial_points


@dataclass(frozen=True)
class MovingViewAssociationConfig:
    maximum_camera_translation_m: float = 0.5
    maximum_camera_rotation_radians: float = math.radians(10)
    maximum_warp_residual_pixels: float = 2.0
    minimum_reference_coverage: float = 0.25
    minimum_axis_span_fraction: float = 0.25
    minimum_correspondences: int = 8

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not isinstance(self.minimum_correspondences, int)
            or self.minimum_correspondences < 8
        ):
            raise ValueError("at least eight original correspondences required")
        if self.minimum_reference_coverage > 1 or self.minimum_axis_span_fraction > 1:
            raise ValueError("coverage fractions must not exceed one")


@dataclass(frozen=True)
class ImageObjectView:
    camera_from_world: np.ndarray
    calibration: object
    config: MovingViewAssociationConfig


def image_object_view(packet, calibration, config):
    """Canonical ground-referenced world used by current object geometry.

    The simulator ego/floor references retain their existing oracle/prior status.
    Camera-from-scan already includes the asynchronous camera/scan pose correction.
    """
    c, s = math.cos(packet.robot_yaw), math.sin(packet.robot_yaw)
    scan_from_world = np.eye(4)
    scan_from_world[:2, :2] = [[c, s], [-s, c]]
    scan_from_world[:2, 3] = -scan_from_world[:2, :2] @ np.asarray(
        packet.robot_position_world_xy
    )
    scan_from_world[2, 3] = packet.floor_height_robot_m
    return ImageObjectView(
        packet.camera_from_scan_robot @ scan_from_world, calibration, config
    )


def compare_moving_view(old, mask, points, view, association_config):
    """Compare only to the original source; no chaining to a shrinking surface."""
    cfg = view.config
    reference = old["points"]
    current_to_old, nearest = cKDTree(reference).query(points)
    old_to_current = cKDTree(points).query(reference)[0]
    distance = association_config.support_distance_m
    current_supported = current_to_old <= distance
    original_supported = old_to_current <= distance
    distinct = len(np.unique(nearest[current_supported]))
    audit = dict(
        object_id=old.get("object_id"),
        original_source_time=old["first"],
        current_maximum_distance_m=float(current_to_old.max()),
        original_maximum_distance_m=float(old_to_current.max()),
        distinct_original_correspondences=distinct,
        original_supported_fraction=float(original_supported.mean()),
        original_unobserved_samples=int((~original_supported).sum()),
        raw_mask_iou=float(
            np.sum(mask & old["mask"]) / max(1, np.sum(mask | old["mask"]))
        ),
        scope="static-prop partial correspondence; unobserved reference points are not current measurements",
        rejections=[],
    )
    reasons = audit["rejections"]
    overlapping = bool(current_supported.any())
    if not current_supported.all():
        reasons.append("current surface exceeds original support distance")
    if distinct < cfg.minimum_correspondences:
        reasons.append("insufficient distinct original correspondences")
    if original_supported.mean() < cfg.minimum_reference_coverage:
        reasons.append("insufficient original surface coverage")
    # Distribution in the original surface's significant principal directions.
    # This rejects a tiny repeated tip, even if it has many densely sampled rays.
    _, _, axes = np.linalg.svd(reference - reference.mean(axis=0), full_matrices=False)
    original_spans = np.ptp(reference @ axes.T, axis=0)
    spans = np.ptp(points @ axes.T, axis=0)
    significant = original_spans > 2 * distance
    ratios = spans[significant] / original_spans[significant]
    audit["principal_span_ratios"] = ratios.tolist()
    if not significant.any() or np.any(ratios < cfg.minimum_axis_span_fraction):
        reasons.append("current support lacks original spatial distribution")
    original_camera = old.get("camera_from_world")
    if original_camera is None:
        reasons.append("no original calibrated camera reference")
        return False, overlapping, audit
    relative = view.camera_from_world @ np.linalg.inv(original_camera)
    translation = float(np.linalg.norm(relative[:3, 3]))
    angle = math.acos(float(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1.0, 1.0)))
    audit.update(camera_translation_m=translation, camera_rotation_radians=angle)
    if (
        translation > cfg.maximum_camera_translation_m
        or angle > cfg.maximum_camera_rotation_radians
    ):
        reasons.append("view change exceeds static-prop scope")
        return False, overlapping, audit
    initial = project_fisheye_polynomial_points(
        reference, camera_from_source=original_camera, calibration=view.calibration
    )
    current = project_fisheye_polynomial_points(
        reference,
        camera_from_source=view.camera_from_world,
        calibration=view.calibration,
    )
    if len(initial.source_indices) != len(reference) or len(
        current.source_indices
    ) != len(reference):
        reasons.append("original support leaves calibrated image")
        return False, overlapping, audit
    source = initial.pixel_xy
    design = np.column_stack((source, np.ones(len(source))))
    mapping, _, rank, _ = np.linalg.lstsq(design, current.pixel_xy, rcond=None)
    singular = np.linalg.svd(source - source.mean(axis=0), compute_uv=False)
    if rank < 3 or singular[-1] / math.sqrt(len(source)) < 1.0:
        reasons.append("original projection lacks two-dimensional support")
        return False, overlapping, audit
    residual = np.linalg.norm(design @ mapping - current.pixel_xy, axis=1)
    audit["maximum_warp_residual_pixels"] = float(residual.max())
    if (
        residual.max() > cfg.maximum_warp_residual_pixels
        or np.linalg.det(mapping[:2]) <= 0
    ):
        reasons.append("original-depth affine warp is not sufficiently supported")
        return False, overlapping, audit
    # Do not accept a conveniently clipped silhouette at the image boundary.
    yy, xx = np.where(old["mask"])
    warped_pixels = np.column_stack((xx, yy, np.ones(len(xx)))) @ mapping
    h, w = mask.shape
    if np.any(warped_pixels < 1) or np.any(warped_pixels >= [w - 1, h - 1]):
        reasons.append("projected original outline touches image boundary")
        return False, overlapping, audit
    warped = cv2.warpAffine(
        old["mask"].astype(np.uint8), mapping.T, (w, h), flags=cv2.INTER_NEAREST
    ).astype(bool)
    iou = float(np.sum(warped & mask) / max(1, np.sum(warped | mask)))
    audit["warped_mask_iou"] = iou
    if iou < association_config.minimum_mask_iou:
        reasons.append("current mask disagrees with reprojected original outline")
    return not reasons, overlapping, audit
