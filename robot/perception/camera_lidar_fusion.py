"""Pure NumPy camera--LiDAR semantic association.

The LiDAR tracker owns obstacle geometry and persistent track identifiers.
This module projects the tracked LiDAR returns into one calibrated camera
image and uses instance masks only to attach semantic labels to those tracks.
It never creates obstacle geometry from camera detections.

Nova Carter's front Hawk camera uses Isaac Sim's ``fisheyePolynomial``
projection.  Its authored polynomial maps image radius to ray angle; point
projection therefore numerically inverts that polynomial.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from robot.perception.semantic_detection import SemanticDetectionFrame
from robot.safety.obstacle_tracking import ObstacleTrackingResult
from robot.safety.semantic_label_provider import (
    SemanticClassLabel,
    SemanticLabelResult,
    SemanticTrackAssignment,
)
from robot.safety.semantic_obstacle import make_unknown_semantic_obstacle


FUSION_LABEL_SOURCE = "yolo_camera_lidar_fusion"


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real value while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    """Return one positive integer while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def _immutable_array(
    value: object,
    *,
    dtype: np.dtype,
    shape_tail: tuple[int, ...],
    name: str,
) -> np.ndarray:
    """Return an immutable contiguous array with a validated trailing shape."""

    result = np.asarray(value, dtype=dtype)
    expected_dimensions = len(shape_tail) + 1
    if result.ndim != expected_dimensions or result.shape[1:] != shape_tail:
        shape_text = "N, " + ", ".join(str(item) for item in shape_tail)
        raise ValueError(f"{name} must have shape ({shape_text}).")
    result = np.array(result, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FThetaCameraCalibration:
    """Isaac Sim polynomial-fisheye calibration for one output image."""

    nominal_width: int
    nominal_height: int
    optical_center_x: float
    optical_center_y: float
    maximum_fov_degrees: float
    polynomial_coefficients: tuple[float, float, float, float, float]
    output_width: int
    output_height: int

    def __post_init__(self) -> None:
        nominal_width = _positive_integer(
            self.nominal_width,
            name="nominal_width",
        )
        nominal_height = _positive_integer(
            self.nominal_height,
            name="nominal_height",
        )
        output_width = _positive_integer(
            self.output_width,
            name="output_width",
        )
        output_height = _positive_integer(
            self.output_height,
            name="output_height",
        )
        center_x = _finite_real(
            self.optical_center_x,
            name="optical_center_x",
        )
        center_y = _finite_real(
            self.optical_center_y,
            name="optical_center_y",
        )
        maximum_fov = _finite_real(
            self.maximum_fov_degrees,
            name="maximum_fov_degrees",
        )
        if not 0.0 < center_x < nominal_width:
            raise ValueError("optical_center_x must lie inside the nominal image.")
        if not 0.0 < center_y < nominal_height:
            raise ValueError("optical_center_y must lie inside the nominal image.")
        if not 0.0 < maximum_fov <= 360.0:
            raise ValueError("maximum_fov_degrees must lie in (0, 360].")
        try:
            coefficients = tuple(
                _finite_real(value, name="polynomial coefficient")
                for value in self.polynomial_coefficients
            )
        except TypeError as error:
            raise TypeError(
                "polynomial_coefficients must be an iterable of five values."
            ) from error
        if len(coefficients) != 5:
            raise ValueError("polynomial_coefficients must contain five values.")
        if coefficients[1] <= 0.0:
            raise ValueError(
                "The polynomial linear coefficient must be greater than zero."
            )
        object.__setattr__(self, "nominal_width", nominal_width)
        object.__setattr__(self, "nominal_height", nominal_height)
        object.__setattr__(self, "output_width", output_width)
        object.__setattr__(self, "output_height", output_height)
        object.__setattr__(self, "optical_center_x", center_x)
        object.__setattr__(self, "optical_center_y", center_y)
        object.__setattr__(self, "maximum_fov_degrees", maximum_fov)
        object.__setattr__(self, "polynomial_coefficients", coefficients)


@dataclass(frozen=True)
class ProjectedLidarPoints:
    """Visible LiDAR returns and their camera-image coordinates."""

    source_indices: np.ndarray
    pixel_xy: np.ndarray
    pixel_indices_xy: np.ndarray
    camera_depths: np.ndarray

    def __post_init__(self) -> None:
        source_indices = np.asarray(self.source_indices, dtype=np.int64)
        if source_indices.ndim != 1:
            raise ValueError("source_indices must have shape (N,).")
        source_indices = np.array(source_indices, copy=True, order="C")
        source_indices.setflags(write=False)
        pixel_xy = _immutable_array(
            self.pixel_xy,
            dtype=np.float64,
            shape_tail=(2,),
            name="pixel_xy",
        )
        pixel_indices_xy = _immutable_array(
            self.pixel_indices_xy,
            dtype=np.int64,
            shape_tail=(2,),
            name="pixel_indices_xy",
        )
        camera_depths = np.asarray(self.camera_depths, dtype=np.float64)
        if camera_depths.ndim != 1:
            raise ValueError("camera_depths must have shape (N,).")
        camera_depths = np.array(camera_depths, copy=True, order="C")
        camera_depths.setflags(write=False)
        lengths = {
            source_indices.shape[0],
            pixel_xy.shape[0],
            pixel_indices_xy.shape[0],
            camera_depths.shape[0],
        }
        if len(lengths) != 1:
            raise ValueError("Projected LiDAR arrays must have equal length.")
        if not np.isfinite(pixel_xy).all() or not np.isfinite(camera_depths).all():
            raise ValueError("Projected LiDAR values must be finite.")
        if np.any(camera_depths <= 0.0):
            raise ValueError("camera_depths must be greater than zero.")
        object.__setattr__(self, "source_indices", source_indices)
        object.__setattr__(self, "pixel_xy", pixel_xy)
        object.__setattr__(self, "pixel_indices_xy", pixel_indices_xy)
        object.__setattr__(self, "camera_depths", camera_depths)


@dataclass(frozen=True)
class CameraLidarFusionConfig:
    """Conservative instance-mask association thresholds."""

    minimum_visible_points: int = 2
    minimum_mask_hits: int = 2
    minimum_overlap_fraction: float = 0.25
    mask_dilation_radius_pixels: int = 3
    ambiguity_score_margin: float = 0.05
    minimum_camera_depth: float = 0.05
    require_projected_median_in_box: bool = True
    projected_median_box_margin_pixels: int = 8

    def __post_init__(self) -> None:
        for name in ("minimum_visible_points", "minimum_mask_hits"):
            _positive_integer(getattr(self, name), name=name)
        dilation = self.mask_dilation_radius_pixels
        if isinstance(dilation, bool) or not isinstance(dilation, Integral):
            raise TypeError("mask_dilation_radius_pixels must be an integer.")
        if dilation < 0:
            raise ValueError("mask_dilation_radius_pixels cannot be negative.")
        overlap = _finite_real(
            self.minimum_overlap_fraction,
            name="minimum_overlap_fraction",
        )
        ambiguity = _finite_real(
            self.ambiguity_score_margin,
            name="ambiguity_score_margin",
        )
        depth = _finite_real(
            self.minimum_camera_depth,
            name="minimum_camera_depth",
        )
        if not 0.0 <= overlap <= 1.0:
            raise ValueError("minimum_overlap_fraction must lie in [0, 1].")
        if not 0.0 <= ambiguity <= 1.0:
            raise ValueError("ambiguity_score_margin must lie in [0, 1].")
        if depth <= 0.0:
            raise ValueError("minimum_camera_depth must be greater than zero.")
        if not isinstance(self.require_projected_median_in_box, bool):
            raise TypeError("require_projected_median_in_box must be a boolean.")
        box_margin = self.projected_median_box_margin_pixels
        if isinstance(box_margin, bool) or not isinstance(box_margin, Integral):
            raise TypeError(
                "projected_median_box_margin_pixels must be an integer."
            )
        if box_margin < 0:
            raise ValueError(
                "projected_median_box_margin_pixels cannot be negative."
            )


@dataclass(frozen=True)
class TrackSemanticAssociation:
    """Semantic result and best pixel-overlap evidence for one LiDAR track.

    ``detection_index`` identifies the matched detection when ``status`` is
    ``matched``. For a rejected association it identifies the best attempted
    detection, allowing thresholds to be diagnosed without attaching a label.
    """

    track_id: int
    status: str
    visible_point_count: int
    detection_index: int | None
    mask_hit_count: int
    overlap_fraction: float
    association_score: float
    projected_median_inside_box: bool
    projected_median_xy: tuple[float, float] | None

    @property
    def is_matched(self) -> bool:
        """Return whether this track received one visual label."""

        return self.status == "matched"


@dataclass(frozen=True)
class CameraLidarFusionResult:
    """Complete projection and track-label association result."""

    projection: ProjectedLidarPoints
    projected_track_ids: np.ndarray
    associations: tuple[TrackSemanticAssociation, ...]
    semantic_labels: SemanticLabelResult
    matched_detection_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        projected_track_ids = np.asarray(
            self.projected_track_ids,
            dtype=np.int64,
        )
        if projected_track_ids.shape != (self.projection.source_indices.size,):
            raise ValueError(
                "projected_track_ids must align with projected LiDAR points."
            )
        projected_track_ids = np.array(
            projected_track_ids,
            copy=True,
            order="C",
        )
        projected_track_ids.setflags(write=False)
        object.__setattr__(self, "projected_track_ids", projected_track_ids)


def project_fisheye_polynomial_points(
    points_source_xyz: np.ndarray,
    *,
    camera_from_source: np.ndarray,
    calibration: FThetaCameraCalibration,
    minimum_camera_depth: float = 0.05,
) -> ProjectedLidarPoints:
    """Project 3D points through Isaac Sim's polynomial-fisheye model."""

    if not isinstance(calibration, FThetaCameraCalibration):
        raise TypeError("calibration must be an FThetaCameraCalibration.")
    points = np.asarray(points_source_xyz, dtype=np.float64)
    transform = np.asarray(camera_from_source, dtype=np.float64)
    minimum_depth = _finite_real(
        minimum_camera_depth,
        name="minimum_camera_depth",
    )
    if minimum_depth <= 0.0:
        raise ValueError("minimum_camera_depth must be greater than zero.")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_source_xyz must have shape (N, 3).")
    if not np.isfinite(points).all():
        raise ValueError("points_source_xyz must contain only finite values.")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("camera_from_source must be a finite (4, 4) matrix.")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError("camera_from_source must be a homogeneous transform.")

    camera_points = points @ transform[:3, :3].T + transform[:3, 3]
    forward_depth = -camera_points[:, 2]
    radial_distance = np.hypot(camera_points[:, 0], camera_points[:, 1])
    theta = np.arctan2(radial_distance, forward_depth)
    maximum_theta = math.radians(calibration.maximum_fov_degrees) * 0.5
    ray_mask = (
        (forward_depth > minimum_depth)
        & (theta <= maximum_theta)
    )
    candidate_indices = np.flatnonzero(ray_mask)
    if candidate_indices.size == 0:
        return _empty_projection()

    candidate_theta = theta[candidate_indices]
    image_radius = _invert_radius_polynomial(
        candidate_theta,
        calibration.polynomial_coefficients,
    )
    candidate_radial = radial_distance[candidate_indices]
    directions_x = np.divide(
        camera_points[candidate_indices, 0],
        candidate_radial,
        out=np.zeros_like(candidate_radial),
        where=candidate_radial > 1e-12,
    )
    directions_y = np.divide(
        camera_points[candidate_indices, 1],
        candidate_radial,
        out=np.zeros_like(candidate_radial),
        where=candidate_radial > 1e-12,
    )
    nominal_x = calibration.optical_center_x + image_radius * directions_x
    nominal_y = calibration.optical_center_y - image_radius * directions_y
    pixel_x = nominal_x * calibration.output_width / calibration.nominal_width
    pixel_y = nominal_y * calibration.output_height / calibration.nominal_height
    pixel_xy = np.column_stack((pixel_x, pixel_y))
    inside = (
        (pixel_x >= 0.0)
        & (pixel_x < calibration.output_width)
        & (pixel_y >= 0.0)
        & (pixel_y < calibration.output_height)
        & np.isfinite(pixel_xy).all(axis=1)
    )
    source_indices = candidate_indices[inside]
    pixel_xy = pixel_xy[inside]
    pixel_indices = np.rint(pixel_xy).astype(np.int64)
    pixel_indices[:, 0] = np.clip(
        pixel_indices[:, 0],
        0,
        calibration.output_width - 1,
    )
    pixel_indices[:, 1] = np.clip(
        pixel_indices[:, 1],
        0,
        calibration.output_height - 1,
    )
    return ProjectedLidarPoints(
        source_indices=source_indices,
        pixel_xy=pixel_xy,
        pixel_indices_xy=pixel_indices,
        camera_depths=forward_depth[source_indices],
    )


class CameraLidarSemanticFusion:
    """Attach YOLO instance labels to existing LiDAR track IDs."""

    def __init__(self, config: CameraLidarFusionConfig | None = None) -> None:
        self.config = config or CameraLidarFusionConfig()

    def fuse(
        self,
        *,
        detection_frame: SemanticDetectionFrame,
        obstacle_points_source_xyz: np.ndarray,
        tracking_result: ObstacleTrackingResult,
        camera_from_source: np.ndarray,
        calibration: FThetaCameraCalibration,
    ) -> CameraLidarFusionResult:
        """Project points, associate masks, and return conservative labels."""

        if not isinstance(detection_frame, SemanticDetectionFrame):
            raise TypeError("detection_frame must be a SemanticDetectionFrame.")
        if not isinstance(tracking_result, ObstacleTrackingResult):
            raise TypeError("tracking_result must be an ObstacleTrackingResult.")
        points = np.asarray(obstacle_points_source_xyz, dtype=np.float64)
        track_ids = np.asarray(tracking_result.point_track_ids, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("obstacle_points_source_xyz must have shape (N, 3).")
        if track_ids.shape != (points.shape[0],):
            raise ValueError(
                "tracking_result.point_track_ids must align with obstacle points."
            )
        if (detection_frame.image_width, detection_frame.image_height) != (
            calibration.output_width,
            calibration.output_height,
        ):
            raise ValueError(
                "Detection-frame dimensions must match calibration output dimensions."
            )

        projection = project_fisheye_polynomial_points(
            points,
            camera_from_source=camera_from_source,
            calibration=calibration,
            minimum_camera_depth=self.config.minimum_camera_depth,
        )
        visible_track_ids = track_ids[projection.source_indices]
        masks = tuple(
            _dilate_mask(
                detection.mask,
                radius=self.config.mask_dilation_radius_pixels,
            )
            for detection in detection_frame.detections
        )
        evidence_by_track: dict[
            int,
            list[tuple[float, int, int, float, bool]],
        ] = {}
        visible_counts: dict[int, int] = {}
        projected_medians: dict[int, tuple[float, float]] = {}
        for track in tracking_result.tracked_obstacles:
            track_mask = visible_track_ids == track.track_id
            visible_count = int(np.count_nonzero(track_mask))
            visible_counts[track.track_id] = visible_count
            if visible_count < self.config.minimum_visible_points:
                continue
            pixels = projection.pixel_indices_xy[track_mask]
            projected_median = np.median(pixels, axis=0)
            projected_medians[track.track_id] = (
                float(projected_median[0]),
                float(projected_median[1]),
            )
            track_evidence: list[tuple[float, int, int, float, bool]] = []
            for detection_index, (detection, mask) in enumerate(
                zip(detection_frame.detections, masks, strict=True)
            ):
                hits = int(np.count_nonzero(mask[pixels[:, 1], pixels[:, 0]]))
                overlap = hits / visible_count
                score = overlap * detection.confidence
                box = detection.bounding_box
                box_margin = self.config.projected_median_box_margin_pixels
                median_inside_box = bool(
                    box.x_min - box_margin
                    <= projected_median[0]
                    <= box.x_max + box_margin
                    and box.y_min - box_margin
                    <= projected_median[1]
                    <= box.y_max + box_margin
                )
                track_evidence.append(
                    (
                        score,
                        detection_index,
                        hits,
                        overlap,
                        median_inside_box,
                    )
                )
            evidence_by_track[track.track_id] = sorted(
                track_evidence,
                reverse=True,
            )

        proposals: list[tuple[float, int, int, int, float]] = []
        pre_status: dict[int, str] = {}
        for track in tracking_result.tracked_obstacles:
            track_id = track.track_id
            visible_count = visible_counts[track_id]
            evidence = evidence_by_track.get(track_id, [])
            if visible_count < self.config.minimum_visible_points:
                pre_status[track_id] = "insufficient_visible_points"
                continue
            candidates = [
                item
                for item in evidence
                if item[2] >= self.config.minimum_mask_hits
                and item[3] >= self.config.minimum_overlap_fraction
                and (
                    item[4]
                    or not self.config.require_projected_median_in_box
                )
            ]
            if not candidates:
                overlap_without_center = any(
                    item[2] >= self.config.minimum_mask_hits
                    and item[3] >= self.config.minimum_overlap_fraction
                    for item in evidence
                )
                pre_status[track_id] = (
                    "projected_median_outside_box"
                    if overlap_without_center
                    else "insufficient_mask_overlap"
                )
                continue
            best_candidate_by_class: dict[
                str,
                tuple[float, int, int, float, bool],
            ] = {}
            for candidate in candidates:
                detection_index = candidate[1]
                semantic_class = detection_frame.detections[
                    detection_index
                ].semantic_class
                best_candidate_by_class.setdefault(semantic_class, candidate)
            ranked_classes = sorted(
                best_candidate_by_class.items(),
                key=lambda item: item[1][0],
                reverse=True,
            )
            if (
                len(ranked_classes) > 1
                and ranked_classes[0][1][0] - ranked_classes[1][1][0]
                < self.config.ambiguity_score_margin
            ):
                pre_status[track_id] = "ambiguous"
                continue
            winning_class = ranked_classes[0][0]
            for score, detection_index, hits, overlap, _ in candidates:
                if (
                    detection_frame.detections[detection_index].semantic_class
                    == winning_class
                ):
                    proposals.append(
                        (score, track_id, detection_index, hits, overlap)
                    )

        selected_by_track: dict[int, tuple[float, int, int, float]] = {}
        used_detection_indices: set[int] = set()
        proposed_track_ids = {proposal[1] for proposal in proposals}
        for score, track_id, detection_index, hits, overlap in sorted(
            proposals,
            reverse=True,
        ):
            if (
                track_id in selected_by_track
                or detection_index in used_detection_indices
            ):
                continue
            used_detection_indices.add(detection_index)
            selected_by_track[track_id] = (
                score,
                detection_index,
                hits,
                overlap,
            )
        for track_id in proposed_track_ids - selected_by_track.keys():
            pre_status[track_id] = "detection_already_assigned"

        tracks_by_id = {
            track.track_id: track for track in tracking_result.tracked_obstacles
        }
        diagnostics: list[TrackSemanticAssociation] = []
        assignments: list[SemanticTrackAssignment] = []
        for track_id in sorted(tracks_by_id):
            selected = selected_by_track.get(track_id)
            if selected is None:
                evidence = evidence_by_track.get(track_id, [])
                best_evidence = evidence[0] if evidence else None
                assignments.append(
                    SemanticTrackAssignment(
                        make_unknown_semantic_obstacle(tracks_by_id[track_id])
                    )
                )
                diagnostics.append(
                    TrackSemanticAssociation(
                        track_id=track_id,
                        status=pre_status.get(track_id, "insufficient_mask_overlap"),
                        visible_point_count=visible_counts.get(track_id, 0),
                        detection_index=(
                            None if best_evidence is None else best_evidence[1]
                        ),
                        mask_hit_count=(
                            0 if best_evidence is None else best_evidence[2]
                        ),
                        overlap_fraction=(
                            0.0 if best_evidence is None else best_evidence[3]
                        ),
                        association_score=(
                            0.0 if best_evidence is None else best_evidence[0]
                        ),
                        projected_median_inside_box=(
                            False if best_evidence is None else best_evidence[4]
                        ),
                        projected_median_xy=projected_medians.get(track_id),
                    )
                )
                continue
            score, detection_index, hits, overlap = selected
            detection = detection_frame.detections[detection_index]
            label = SemanticClassLabel(
                semantic_class=detection.semantic_class,
                confidence=score,
                source=FUSION_LABEL_SOURCE,
            )
            assignments.append(
                SemanticTrackAssignment(label.attach(tracks_by_id[track_id]))
            )
            diagnostics.append(
                TrackSemanticAssociation(
                    track_id=track_id,
                    status="matched",
                    visible_point_count=visible_counts[track_id],
                    detection_index=detection_index,
                    mask_hit_count=hits,
                    overlap_fraction=overlap,
                    association_score=score,
                    projected_median_inside_box=True,
                    projected_median_xy=projected_medians[track_id],
                )
            )

        return CameraLidarFusionResult(
            projection=projection,
            projected_track_ids=visible_track_ids,
            associations=tuple(diagnostics),
            semantic_labels=SemanticLabelResult(
                assignments=tuple(assignments),
                unmatched_label_count=(
                    len(detection_frame.detections) - len(used_detection_indices)
                ),
            ),
            matched_detection_indices=tuple(sorted(used_detection_indices)),
        )


def _invert_radius_polynomial(
    theta: np.ndarray,
    coefficients: tuple[float, float, float, float, float],
) -> np.ndarray:
    """Invert ``theta = a + b*r + c*r^2 + d*r^3 + e*r^4``."""

    a, b, c, d, e = coefficients
    radius = np.maximum((theta - a) / b, 0.0)
    for _ in range(12):
        value = a + radius * (b + radius * (c + radius * (d + radius * e)))
        derivative = b + radius * (2.0 * c + radius * (3.0 * d + 4.0 * e * radius))
        safe_derivative = np.where(np.abs(derivative) > 1e-12, derivative, b)
        radius = np.maximum(radius - (value - theta) / safe_derivative, 0.0)
    return radius


def _dilate_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
    """Dilate a boolean mask with a small square kernel using NumPy only."""

    source = np.asarray(mask, dtype=np.bool_)
    if radius == 0:
        return source
    height, width = source.shape
    padded = np.pad(source, radius, mode="constant", constant_values=False)
    result = np.zeros_like(source)
    for y_offset in range(2 * radius + 1):
        for x_offset in range(2 * radius + 1):
            result |= padded[
                y_offset : y_offset + height,
                x_offset : x_offset + width,
            ]
    return result


def _empty_projection() -> ProjectedLidarPoints:
    """Return a correctly shaped empty projection result."""

    return ProjectedLidarPoints(
        source_indices=np.empty((0,), dtype=np.int64),
        pixel_xy=np.empty((0, 2), dtype=np.float64),
        pixel_indices_xy=np.empty((0, 2), dtype=np.int64),
        camera_depths=np.empty((0,), dtype=np.float64),
    )
