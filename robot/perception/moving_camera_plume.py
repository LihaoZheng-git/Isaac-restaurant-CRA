"""Registered clean-reference RGB plume proxy for bounded moving-camera views.

Background feature flow compensates camera image motion. A separately validated
clean reference is warped into each current view before extracting plume cues.
Parallax, disocclusion, opaque lookalikes and weak plumes remain limitations.
"""

from collections import deque
from dataclasses import dataclass, replace
import math
from types import MappingProxyType

import cv2
import numpy as np

from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.continuous_thermal_score import estimate_continuous_thermal_score
from robot.perception.object_plume_evidence import ObjectPlumeEvidence
from robot.perception.physical_attributes import (
    CategoricalEstimate,
    EstimateStatus,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.perception.plume_features import (
    PlumeSourceObservation,
    extract_continuous_plume_features,
)
from robot.perception.rgb_plume import RgbPlumeConfig, StationaryRgbPlumeEstimator
from robot.perception.semantic_detection import SemanticDetectionFrame


MOVING_PLUME_SOURCE = "registered_rgb_object_plume_proxy"


@dataclass(frozen=True)
class MovingCameraPlumeConfig:
    minimum_features: int = 40
    minimum_inliers: int = 24
    minimum_local_inliers: int = 6
    minimum_inlier_fraction: float = 0.65
    minimum_background_coverage: float = 0.15
    maximum_reprojection_error_pixels: float = 1.5
    maximum_background_change: float = 8.0
    minimum_brightness_change: float = 12.0
    maximum_chroma_fraction: float = 0.35
    minimum_component_pixels: int = 20
    confidence_cap: float = 0.55
    minimum_pair_seconds: float = 0.05
    maximum_pair_seconds: float = 0.20
    maximum_track_age_seconds: float = 0.15
    minimum_source_pixels: int = 200
    minimum_source_confidence: float = 0.5
    minimum_pairs: int = 3
    maximum_window_age_seconds: float = 0.5
    minimum_registered_source_iou: float = 0.5
    maximum_reference_age_seconds: float = 60.0
    minimum_clean_air_coverage: float = 0.65

    def __post_init__(self):
        integers = (
            "minimum_features",
            "minimum_inliers",
            "minimum_local_inliers",
            "minimum_component_pixels",
            "minimum_source_pixels",
            "minimum_pairs",
        )
        for name in integers:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in self.__dataclass_fields__:
            if name in integers:
                continue
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "minimum_inlier_fraction",
            "minimum_background_coverage",
            "maximum_chroma_fraction",
            "confidence_cap",
            "minimum_source_confidence",
            "minimum_registered_source_iou",
            "minimum_clean_air_coverage",
        ):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must lie in (0, 1]")
        if (
            self.minimum_pairs < 3
            or self.maximum_pair_seconds < self.minimum_pair_seconds
        ):
            raise ValueError(
                "at least three pairs and a valid temporal interval are required"
            )


@dataclass(frozen=True)
class TrackedRgbObjectFrame:
    rgb: np.ndarray
    detections: SemanticDetectionFrame
    association: TrackSemanticAssociation
    track_timestamp_seconds: float

    def __post_init__(self):
        frame = self.detections
        image = np.array(self.rgb, copy=True)
        if image.dtype != np.uint8 or image.shape != (
            frame.image_height,
            frame.image_width,
            3,
        ):
            raise ValueError("RGB must align with the source detection frame")
        if not isinstance(self.association, TrackSemanticAssociation):
            raise TypeError("an explicit track association is required")
        if (
            not math.isfinite(self.track_timestamp_seconds)
            or self.track_timestamp_seconds < 0
        ):
            raise ValueError("track timestamp must be finite and nonnegative")
        image.setflags(write=False)
        object.__setattr__(self, "rgb", image)

    @property
    def detection(self):
        index = self.association.detection_index
        if (
            not self.association.is_matched
            or index is None
            or not 0 <= index < len(self.detections.detections)
        ):
            return None
        return self.detections.detections[index]


@dataclass(frozen=True)
class BackgroundRegistration:
    homography: np.ndarray | None
    rejection_reasons: tuple[str, ...]
    diagnostics: dict


def _cue(value, timestamp, confidence, detail):
    return ScalarEstimate(
        value,
        max(0.0, value - 0.15),
        min(1.0, value + 0.15),
        confidence,
        "1",
        MOVING_PLUME_SOURCE,
        timestamp,
        EstimateStatus.INFERRED,
        (detail,),
    )


def _region(image, detection):
    return StationaryRgbPlumeEstimator(
        image,
        source_box=detection.bounding_box,
        config=RgbPlumeConfig(source_height_multiplier=1.5),
    ).roi


def register_background_pair(previous, current, *, config=None):
    settings = config or MovingCameraPlumeConfig()
    timestamp = current.detections.source_timestamp_seconds
    diagnostics = {}

    def reject(reason):
        return BackgroundRegistration(None, (reason,), diagnostics)

    old, new = previous.detection, current.detection
    if (
        old is None
        or new is None
        or previous.association.track_id != current.association.track_id
    ):
        return reject("source-track association is missing or changed")
    if (
        previous.rgb.shape != current.rgb.shape
        or current.detections.source_frame_id <= previous.detections.source_frame_id
    ):
        return reject("source frames must align and have increasing IDs")
    delta = timestamp - previous.detections.source_timestamp_seconds
    if (
        not settings.minimum_pair_seconds
        <= delta
        <= settings.maximum_pair_seconds + 1e-9
    ):
        return reject("RGB pair interval is outside the validated temporal range")
    for item in (previous, current):
        detection = item.detection
        age = item.detections.source_timestamp_seconds - item.track_timestamp_seconds
        mask = detection.mask
        if not 0 <= age <= settings.maximum_track_age_seconds + 1e-9:
            return reject("source track is stale or future for the RGB frame")
        if (
            detection.confidence < settings.minimum_source_confidence
            or mask.sum() < settings.minimum_source_pixels
            or mask[:2].any()
            or mask[-2:].any()
            or mask[:, :2].any()
            or mask[:, -2:].any()
        ):
            return reject("source mask is unresolved, unconfident or truncated")
        box = detection.bounding_box
        if (
            min(box.y_min, 1.5 * box.height) / (1.5 * box.height)
            < settings.minimum_clean_air_coverage
        ):
            return reject("insufficient visible air region above the source")
    try:
        old_roi, roi = _region(previous.rgb, old), _region(current.rgb, new)
    except ValueError as error:
        return reject(str(error))
    previous_background = ~(
        old_roi
        | cv2.dilate(old.mask.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(
            bool
        )
    )
    background = ~(
        roi
        | cv2.dilate(new.mask.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(
            bool
        )
    )
    grey_old = cv2.cvtColor(previous.rgb, cv2.COLOR_RGB2GRAY)
    grey = cv2.cvtColor(current.rgb, cv2.COLOR_RGB2GRAY)
    points = cv2.goodFeaturesToTrack(
        grey_old, 800, 0.01, 5, mask=(previous_background.astype(np.uint8) * 255)
    )
    if points is None or len(points) < settings.minimum_features:
        return reject("insufficient independent background features")
    moved, ok, _ = cv2.calcOpticalFlowPyrLK(
        grey_old, grey, points, None, winSize=(21, 21), maxLevel=3
    )
    if moved is None:
        return reject("background optical flow failed")
    back, back_ok, _ = cv2.calcOpticalFlowPyrLK(
        grey, grey_old, moved, None, winSize=(21, 21), maxLevel=3
    )
    if back is None:
        return reject("background backward optical flow failed")
    xy, next_xy = points[:, 0], moved[:, 0]
    height, width = grey.shape
    pixel = np.rint(next_xy).astype(int)
    good = (
        ok[:, 0].astype(bool)
        & back_ok[:, 0].astype(bool)
        & (np.linalg.norm(back[:, 0] - xy, axis=1) <= 1.0)
        & (pixel[:, 0] >= 0)
        & (pixel[:, 0] < width)
        & (pixel[:, 1] >= 0)
        & (pixel[:, 1] < height)
    )
    good &= background[
        np.clip(pixel[:, 1], 0, height - 1), np.clip(pixel[:, 0], 0, width - 1)
    ]
    if good.sum() < settings.minimum_features:
        return reject("insufficient consistent background flow")
    homography, inliers = cv2.findHomography(
        xy[good], next_xy[good], cv2.RANSAC, settings.maximum_reprojection_error_pixels
    )
    if homography is None or inliers is None or not np.isfinite(homography).all():
        return reject("background registration failed")
    accepted = inliers[:, 0].astype(bool)
    diagnostics.update(
        feature_count=int(good.sum()),
        inlier_count=int(accepted.sum()),
        inlier_fraction=float(accepted.mean()),
    )
    if (
        accepted.sum() < settings.minimum_inliers
        or accepted.mean() < settings.minimum_inlier_fraction
    ):
        return reject("background registration quality is insufficient")
    coverage = cv2.contourArea(
        cv2.convexHull(next_xy[good][accepted].astype(np.float32))
    ) / (height * width)
    diagnostics["inlier_image_coverage"] = float(coverage)
    if coverage < settings.minimum_background_coverage:
        return reject("independent background features cover too little of the image")
    box = new.bounding_box
    local = next_xy[good][accepted]
    near = (
        (local[:, 0] >= box.x_min - 2 * box.width)
        & (local[:, 0] <= box.x_max + 2 * box.width)
        & (local[:, 1] >= box.y_min - 3 * box.height)
        & (local[:, 1] <= box.y_max)
    )
    diagnostics["local_inlier_count"] = int(near.sum())
    if near.sum() < settings.minimum_local_inliers:
        return reject("source vicinity lacks independent registration support")
    corners = np.array(
        [
            [
                [old.bounding_box.x_min, old.bounding_box.y_min],
                [old.bounding_box.x_max, old.bounding_box.y_min],
                [old.bounding_box.x_max, old.bounding_box.y_max],
                [old.bounding_box.x_min, old.bounding_box.y_max],
            ]
        ],
        np.float32,
    )
    warped_box = cv2.perspectiveTransform(corners, homography)[0]
    x0, y0 = warped_box.min(axis=0)
    x1, y1 = warped_box.max(axis=0)
    intersection = max(0.0, min(x1, box.x_max) - max(x0, box.x_min)) * max(
        0.0, min(y1, box.y_max) - max(y0, box.y_min)
    )
    iou = intersection / max(1.0, (x1 - x0) * (y1 - y0) + box.area - intersection)
    diagnostics["registered_source_iou"] = float(iou)
    if iou < settings.minimum_registered_source_iou:
        return reject("registered source does not match the current instance")
    warped = cv2.warpPerspective(previous.rgb, homography, (width, height))
    valid = cv2.warpPerspective(
        np.ones((height, width), np.uint8),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    if np.mean(valid[roi]) < 0.95:
        return reject("plume region is truncated by registration")
    checked_background = valid & background
    if checked_background.mean() < 0.10:
        return reject("insufficient independent registered background")
    difference = current.rgb.astype(float) - warped.astype(float)
    change = float(np.abs(difference[checked_background]).mean())
    diagnostics["background_change"] = change
    if change > settings.maximum_background_change:
        return reject("background mismatch or lighting change invalidates registration")
    diagnostics["homography"] = homography.tolist()
    homography.setflags(write=False)
    return BackgroundRegistration(homography, (), diagnostics)


class MovingCameraObjectPlumeEstimator:
    def __init__(self, config=None):
        self.config = config or MovingCameraPlumeConfig()
        self.previous = {}
        self.windows = {}
        self.last_diagnostics = {}
        self.references = {}
        self.homographies = {}

    def update_many(self, observations, *, negative_validation_scope=None):
        observations = tuple(observations)
        tracks = [item.association.track_id for item in observations]
        if len(set(tracks)) != len(tracks):
            raise ValueError("a batch must have distinct track IDs")
        epochs = {
            (item.detections.source_frame_id, item.detections.source_timestamp_seconds)
            for item in observations
        }
        if len(epochs) > 1:
            raise ValueError("a plume batch must use the same RGB epoch")
        if observations and any(
            not np.array_equal(item.rgb, observations[0].rgb)
            for item in observations[1:]
        ):
            raise ValueError("a plume batch must use the same camera image")
        rows = {
            item.association.track_id: self.update(
                item, negative_validation_scope=negative_validation_scope
            )
            for item in observations
        }

        def reject(track, reason):
            row = rows[track]
            timestamp = row.thermal.heat_source_likelihood.timestamp_seconds
            base = PhysicalAttributes.unknown(
                track_id=track, timestamp_seconds=timestamp
            ).thermal
            rows[track] = replace(
                row,
                thermal=replace(
                    base,
                    heat_source_likelihood=ScalarEstimate.unavailable(
                        unit="1",
                        source=MOVING_PLUME_SOURCE,
                        timestamp_seconds=timestamp,
                        evidence=(reason,),
                    ),
                ),
                rejection_reasons=(reason,),
            )

        for index, first in enumerate(observations):
            for second in observations[index + 1 :]:
                a, b = first.association.track_id, second.association.track_id
                if (
                    first.association.is_matched
                    and second.association.is_matched
                    and first.association.detection_index
                    == second.association.detection_index
                ):
                    reject(a, "one source mask is claimed by multiple tracks")
                    reject(b, "one source mask is claimed by multiple tracks")
                    continue
                if not (
                    rows[a].thermal.heat_source_likelihood.is_available
                    and rows[b].thermal.heat_source_likelihood.is_available
                ):
                    continue
                fa, fb = self.windows[a][-1].frame, self.windows[b][-1].frame
                if not fa.instances or not fb.instances:
                    continue
                ma, mb = fa.instances[0].mask, fb.instances[0].mask
                shared = np.count_nonzero(ma & mb) / max(1, min(ma.sum(), mb.sum()))
                if shared > 0.5:
                    reject(
                        a,
                        "positive plume foreground is shared by multiple source regions",
                    )
                    reject(
                        b,
                        "positive plume foreground is shared by multiple source regions",
                    )
        return MappingProxyType(rows)

    def set_reference(self, observation, *, clean_reference_validated):
        if clean_reference_validated is not True:
            raise ValueError("a separately validated clean reference is required")
        if observation.detection is None:
            raise ValueError("clean reference requires an accepted object instance")
        track = observation.association.track_id
        self.references[track] = observation
        self.previous[track] = observation
        self.homographies[track] = np.eye(3)
        self.windows.pop(track, None)

    def update(
        self, observation: TrackedRgbObjectFrame, *, negative_validation_scope=None
    ):
        track = observation.association.track_id
        timestamp = observation.detections.source_timestamp_seconds

        def unknown(reason):
            base = PhysicalAttributes.unknown(
                track_id=track, timestamp_seconds=timestamp
            ).thermal
            return ObjectPlumeEvidence(
                track,
                replace(
                    base,
                    heat_source_likelihood=ScalarEstimate.unavailable(
                        unit="1",
                        source=MOVING_PLUME_SOURCE,
                        timestamp_seconds=timestamp,
                        evidence=(reason,),
                    ),
                ),
                0,
                0,
                (reason,),
                negative_validation_scope,
            )

        if negative_validation_scope is not None and (
            not isinstance(negative_validation_scope, str)
            or not negative_validation_scope.strip()
        ):
            raise ValueError("negative validation scope must be explicit nonempty text")
        previous = self.previous.get(track)
        if previous is not None and (
            observation.detections.source_frame_id
            <= previous.detections.source_frame_id
            or timestamp <= previous.detections.source_timestamp_seconds
        ):
            return unknown("repeated or out-of-order RGB epoch")
        reference = self.references.get(track)
        if previous is None or reference is None:
            return unknown("validated clean reference is unavailable")
        if (
            timestamp - reference.detections.source_timestamp_seconds
            > self.config.maximum_reference_age_seconds
        ):
            return unknown("clean reference exceeds its declared temporal domain")
        result = register_background_pair(previous, observation, config=self.config)
        self.last_diagnostics[track] = result.diagnostics
        if result.homography is None:
            self.windows.pop(track, None)
            return unknown("; ".join(result.rejection_reasons))
        homography = result.homography @ self.homographies[track]
        height, width = observation.rgb.shape[:2]
        warped = cv2.warpPerspective(reference.rgb, homography, (width, height))
        valid = cv2.warpPerspective(
            np.ones((height, width), np.uint8),
            homography,
            (width, height),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)
        old_body = cv2.warpPerspective(
            reference.detection.mask.astype(np.uint8),
            homography,
            (width, height),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)
        old_body = cv2.dilate(
            old_body.astype(np.uint8), np.ones((11, 11), np.uint8)
        ).astype(bool)
        detection = observation.detection
        roi = _region(observation.rgb, detection)
        usable = roi & valid & ~old_body
        coverage = np.count_nonzero(usable) / max(1, np.count_nonzero(roi))
        self.last_diagnostics[track]["clean_air_coverage"] = coverage
        self.last_diagnostics[track]["reference_frame_id"] = (
            reference.detections.source_frame_id
        )
        if coverage < self.config.minimum_clean_air_coverage:
            self.windows.pop(track, None)
            return unknown(
                "registered clean reference cannot cover enough current air region"
            )
        exclusion = old_body | detection.mask | ~valid
        try:
            backend = StationaryRgbPlumeEstimator(
                warped,
                source_box=detection.bounding_box,
                config=RgbPlumeConfig(
                    source_height_multiplier=1.5,
                    minimum_brightness_change=self.config.minimum_brightness_change,
                    maximum_chroma_fraction=self.config.maximum_chroma_fraction,
                    minimum_component_pixels=self.config.minimum_component_pixels,
                    maximum_background_change=self.config.maximum_background_change,
                    confidence_cap=self.config.confidence_cap,
                ),
                background_exclusion_mask=exclusion,
            )
        except ValueError as error:
            return unknown(str(error))
        backend.roi &= usable
        absolute = backend.estimate(
            observation.rgb,
            source_frame_id=observation.detections.source_frame_id,
            timestamp_seconds=timestamp,
        )
        if absolute.frame is None:
            self.windows.pop(track, None)
            return unknown(
                "registered clean reference mismatch: "
                + "; ".join(absolute.rejection_reasons)
            )
        self.previous[track] = observation
        self.homographies[track] = homography
        result = replace(
            absolute,
            frame=replace(
                absolute.frame,
                model_name=MOVING_PLUME_SOURCE,
                model_source="OpenCV background registration plus validated clean-reference subtraction; no trained weights",
            ),
            spatial_association=replace(
                absolute.spatial_association, source=MOVING_PLUME_SOURCE
            ),
        )
        self.last_diagnostics[track]["plume_pixels"] = sum(
            item.mask_pixel_count for item in result.frame.instances
        )
        confidence = min(detection.confidence, self.config.confidence_cap)
        source = _cue(
            1.0, timestamp, confidence, "matched object region, not a hot-object class"
        )
        item = PlumeSourceObservation(
            result.frame,
            detection.bounding_box,
            source,
            result.spatial_association if result.frame.instances else source,
            0 if result.frame.instances else None,
            negative_observation_validated=negative_validation_scope is not None,
            negative_observation_confidence=confidence
            if negative_validation_scope is not None
            else 0.0,
        )
        window = self.windows.setdefault(track, deque(maxlen=self.config.minimum_pairs))
        window.append(item)
        if len(window) < self.config.minimum_pairs:
            return unknown("insufficient accepted registered plume pairs")
        if (
            timestamp - window[0].frame.source_timestamp_seconds
            > self.config.maximum_window_age_seconds
        ):
            return unknown("registered plume window is stale")
        features = extract_continuous_plume_features(
            tuple(window), timestamp_seconds=timestamp
        )
        score = estimate_continuous_thermal_score(features, timestamp_seconds=timestamp)
        heat = replace(
            score.thermal.heat_source_likelihood,
            source=MOVING_PLUME_SOURCE,
            evidence=(
                *score.thermal.heat_source_likelihood.evidence,
                "registered clean-reference plume proxy; weak plume, parallax and lookalikes remain limitations",
                f"clean_reference_frame_id={reference.detections.source_frame_id}; timestamp={reference.detections.source_timestamp_seconds}",
                f"negative_validation_scope={negative_validation_scope}",
            ),
        )
        thermal = replace(
            score.thermal,
            heat_source_likelihood=heat,
            source_type=CategoricalEstimate(
                MOVING_PLUME_SOURCE,
                heat.confidence,
                MOVING_PLUME_SOURCE,
                timestamp,
                heat.status,
                heat.evidence,
            )
            if heat.is_available
            else score.thermal.source_type,
        )
        return ObjectPlumeEvidence(
            track,
            thermal,
            len(window),
            features.positive_frame_count,
            score.rejection_reasons,
            negative_validation_scope,
        )
