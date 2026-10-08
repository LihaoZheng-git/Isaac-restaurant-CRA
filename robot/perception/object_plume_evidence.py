"""Object-level visible-plume cues with validated low-cue observations.

This is a bounded fixed-camera reference method, independent of object class.
It estimates visible plume evidence, not temperature, steam identity or soup.
A clean reference and domain validation are explicit prerequisites; copying a
first frame or a failed detector response is not accepted negative evidence.
"""

from dataclasses import dataclass, replace
import math
from types import MappingProxyType

import numpy as np

from robot.perception.continuous_thermal_score import estimate_continuous_thermal_score
from robot.perception.physical_attributes import (
    CategoricalEstimate,
    EstimateStatus,
    PhysicalAttributes,
    ScalarEstimate,
    ThermalAttributes,
)
from robot.perception.plume_features import (
    PlumeSourceObservation,
    extract_continuous_plume_features,
)
from robot.perception.rgb_plume import RgbPlumeConfig, StationaryRgbPlumeEstimator
from robot.perception.semantic_detection import SemanticDetectionFrame
from robot.perception.specialist_track_evidence import SpecialistAssociationFrame


OBJECT_PLUME_SOURCE = "object_visible_plume_proxy"
OBJECT_PLUME_BASIS = "visual_object_plume_proxy"


@dataclass(frozen=True)
class ObjectPlumeObservation:
    rgb: np.ndarray
    detections: SemanticDetectionFrame
    associations: SpecialistAssociationFrame
    track_timestamp_seconds: float
    camera_stationary: bool
    region_status: EstimateStatus = EstimateStatus.INFERRED

    def __post_init__(self):
        image = np.asarray(self.rgb)
        frame, links = self.detections, self.associations
        if not isinstance(frame, SemanticDetectionFrame) or not isinstance(
            links, SpecialistAssociationFrame
        ):
            raise TypeError("object plume requires detection and association frames")
        if image.dtype != np.uint8 or image.shape != (
            frame.image_height,
            frame.image_width,
            3,
        ):
            raise ValueError("RGB must match the detection frame as HxWx3 uint8")
        if (
            frame.source_frame_id != links.source_frame_id
            or frame.source_timestamp_seconds != links.source_timestamp_seconds
            or frame.image_height != links.image_height
            or frame.image_width != links.image_width
        ):
            raise ValueError(
                "object plume frame IDs, timestamps and dimensions must align"
            )
        if (
            not math.isfinite(self.track_timestamp_seconds)
            or self.track_timestamp_seconds < 0
        ):
            raise ValueError("track timestamp must be finite and nonnegative")
        if not isinstance(self.camera_stationary, bool):
            raise TypeError("camera_stationary must be bool")
        status = EstimateStatus(self.region_status)
        if status is EstimateStatus.UNAVAILABLE:
            raise ValueError("observed object region cannot have unavailable status")
        copy = np.array(image, copy=True)
        copy.setflags(write=False)
        object.__setattr__(self, "rgb", copy)
        object.__setattr__(self, "region_status", status)


@dataclass(frozen=True)
class ObjectPlumeConfig:
    minimum_frames: int = 3
    minimum_window_seconds: float = 0.15
    maximum_window_age_seconds: float = 0.5
    maximum_track_age_seconds: float = 0.5
    minimum_source_pixels: int = 200
    minimum_detection_confidence: float = 0.5
    minimum_air_region_fraction: float = 0.65
    maximum_box_shift_fraction: float = 0.10
    negative_observation_confidence: float = 0.55

    def __post_init__(self):
        for name in ("minimum_frames", "minimum_source_pixels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_frames < 3:
            raise ValueError("at least three frames are required")
        for name in self.__dataclass_fields__:
            if name in {"minimum_frames", "minimum_source_pixels"}:
                continue
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            if name.endswith(("fraction", "confidence")) and value > 1:
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.minimum_window_seconds > self.maximum_window_age_seconds:
            raise ValueError("minimum window exceeds maximum evidence age")


@dataclass(frozen=True)
class ObjectPlumeEvidence:
    track_id: int
    thermal: ThermalAttributes
    invoked_frames: int
    positive_frames: int
    rejection_reasons: tuple[str, ...]
    validation_scope: str | None


def estimate_object_plume_evidence(
    reference_rgb: np.ndarray,
    *,
    reference_timestamp_seconds: float,
    observations: tuple[ObjectPlumeObservation, ...],
    timestamp_seconds: float,
    clean_reference_validated: bool = False,
    negative_validation_scope: str | None = None,
    config: ObjectPlumeConfig | None = None,
):
    """Return a result for every track, including explicit unknown records.

    Regions come from matched object masks or explicitly labeled fixture priors.
    Object names never imply heat, an open container, or negative evidence.
    The clean reference must precede distinct, stationary, fresh observations.
    """
    settings = config or ObjectPlumeConfig()
    if not isinstance(settings, ObjectPlumeConfig):
        raise TypeError("config must be ObjectPlumeConfig")
    timestamp = float(timestamp_seconds)
    reference_time = float(reference_timestamp_seconds)
    if not all(math.isfinite(t) and t >= 0 for t in (timestamp, reference_time)):
        raise ValueError("timestamps must be finite and nonnegative")
    if not isinstance(clean_reference_validated, bool):
        raise TypeError("clean_reference_validated must be bool")
    if negative_validation_scope is not None and (
        not isinstance(negative_validation_scope, str)
        or not negative_validation_scope.strip()
    ):
        raise ValueError("negative validation scope must be nonempty text or None")
    items = tuple(observations)
    if any(not isinstance(item, ObjectPlumeObservation) for item in items):
        raise TypeError("observations must contain ObjectPlumeObservation")
    tracks = sorted(
        {link.track_id for item in items for link in item.associations.associations}
    )

    def unknown(track_id, reasons, invoked=0):
        thermal = PhysicalAttributes.unknown(
            track_id=track_id, timestamp_seconds=timestamp
        ).thermal
        cue = ScalarEstimate.unavailable(
            unit="1",
            source=OBJECT_PLUME_SOURCE,
            timestamp_seconds=timestamp,
            evidence=tuple(reasons),
        )
        return ObjectPlumeEvidence(
            track_id,
            replace(thermal, heat_source_likelihood=cue),
            invoked,
            0,
            tuple(reasons),
            negative_validation_scope,
        )

    global_reasons = []
    if not clean_reference_validated:
        global_reasons.append("clean reference has not been validated")
    if len(items) < settings.minimum_frames:
        global_reasons.append("insufficient observation frames")
    if items:
        times = [item.detections.source_timestamp_seconds for item in items]
        ids = [item.detections.source_frame_id for item in items]
        if len(set(ids)) != len(ids) or any(b <= a for a, b in zip(times, times[1:])):
            global_reasons.append(
                "observation IDs/timestamps must be distinct and increasing"
            )
        if times[0] <= reference_time or times[-1] > timestamp:
            global_reasons.append(
                "reference must precede observations; future observations rejected"
            )
        if timestamp - times[0] > settings.maximum_window_age_seconds:
            global_reasons.append("observation window is stale")
        if times[-1] - times[0] < settings.minimum_window_seconds:
            global_reasons.append("temporal observation span is insufficient")
        if any(not item.camera_stationary for item in items):
            global_reasons.append(
                "moving camera is unsupported by fixed-reference plume estimator"
            )
    if global_reasons:
        return MappingProxyType(
            {track: unknown(track, global_reasons) for track in tracks}
        )
    reference = np.asarray(reference_rgb)
    if (
        reference.dtype != np.uint8
        or not items
        or reference.shape != items[0].rgb.shape
    ):
        raise ValueError("reference must match observed uint8 RGB dimensions")

    # Other observed source regions may change legitimately. Exclude their
    # search columns from the common background check, while retaining enough
    # independent pixels to detect camera/lighting changes.
    background_exclusion = np.zeros(reference.shape[:2], bool)
    backend_config = RgbPlumeConfig(source_height_multiplier=1.5)
    for detection in items[0].detections.detections:
        if (
            detection.confidence < settings.minimum_detection_confidence
            or detection.mask_pixel_count < settings.minimum_source_pixels
        ):
            continue
        try:
            region = StationaryRgbPlumeEstimator(
                reference, source_box=detection.bounding_box, config=backend_config
            ).roi
        except ValueError:
            continue
        background_exclusion |= region
    outputs, masks_by_track = {}, {}
    for track in tracks:
        selected = []
        reasons = []
        for item in items:
            links = [
                link
                for link in item.associations.associations
                if link.track_id == track
            ]
            if len(links) != 1 or not links[0].is_matched:
                reasons.append("object-track association missing or ambiguous")
                break
            link = links[0]
            if (
                sum(
                    other.is_matched and other.detection_index == link.detection_index
                    for other in item.associations.associations
                )
                != 1
            ):
                reasons.append("object mask is associated with multiple tracks")
                break
            index = link.detection_index
            if index is None or not 0 <= index < len(item.detections.detections):
                raise ValueError("matched object detection index is invalid")
            detection = item.detections.detections[index]
            age = (
                item.detections.source_timestamp_seconds - item.track_timestamp_seconds
            )
            if age < 0 or age > settings.maximum_track_age_seconds:
                reasons.append(
                    "track geometry is future or stale for camera association"
                )
            if (
                detection.mask_pixel_count < settings.minimum_source_pixels
                or detection.confidence < settings.minimum_detection_confidence
            ):
                reasons.append("object region lacks resolved/confident observation")
            if (
                detection.mask[:2].any()
                or detection.mask[-2:].any()
                or detection.mask[:, :2].any()
                or detection.mask[:, -2:].any()
            ):
                reasons.append("object is truncated at image border")
            box = detection.bounding_box
            if (
                min(box.y_min, 1.5 * box.height) / (1.5 * box.height)
                < settings.minimum_air_region_fraction
            ):
                reasons.append("insufficient visible air region above object")
            selected.append((item, detection))
        if reasons:
            outputs[track] = unknown(track, tuple(dict.fromkeys(reasons)))
            continue
        first_box = selected[0][1].bounding_box
        try:
            backend = StationaryRgbPlumeEstimator(
                reference,
                source_box=first_box,
                config=backend_config,
                background_exclusion_mask=background_exclusion,
            )
        except ValueError as error:
            outputs[track] = unknown(track, (str(error),))
            continue
        window, masks = [], []
        for item, detection in selected:
            box = detection.bounding_box
            displacement = max(
                abs(getattr(box, name) - getattr(first_box, name))
                for name in ("x_min", "y_min", "x_max", "y_max")
            )
            if displacement > settings.maximum_box_shift_fraction * min(
                first_box.width, first_box.height
            ):
                reasons.append("source moved or object region is unstable")
                break
            frame = item.detections
            result = backend.estimate(
                item.rgb,
                source_frame_id=frame.source_frame_id,
                timestamp_seconds=frame.source_timestamp_seconds,
            )
            if result.frame is None:
                reasons.extend(result.rejection_reasons)
                break
            confidence = min(
                detection.confidence, settings.negative_observation_confidence
            )
            observed_source = ScalarEstimate(
                1.0,
                1.0,
                1.0,
                confidence,
                "1",
                "observed_object_region",
                frame.source_timestamp_seconds,
                item.region_status,
                (
                    "object region only; no container/hot-soup classification or material inference",
                ),
            )
            positive = bool(result.frame.instances)
            relation = result.spatial_association if positive else observed_source
            window.append(
                PlumeSourceObservation(
                    result.frame,
                    first_box,
                    observed_source,
                    relation,
                    0 if positive else None,
                    negative_observation_validated=negative_validation_scope
                    is not None,
                    negative_observation_confidence=confidence
                    if negative_validation_scope is not None
                    else 0.0,
                )
            )
            masks.append(
                result.frame.instances[0].mask
                if positive
                else np.zeros(reference.shape[:2], bool)
            )
        if reasons:
            outputs[track] = unknown(track, tuple(dict.fromkeys(reasons)), len(window))
            continue
        features = extract_continuous_plume_features(
            tuple(window), timestamp_seconds=timestamp
        )
        score = estimate_continuous_thermal_score(features, timestamp_seconds=timestamp)
        heat = replace(
            score.thermal.heat_source_likelihood,
            source=OBJECT_PLUME_SOURCE,
            evidence=score.thermal.heat_source_likelihood.evidence
            + (
                "object-level visible-plume proxy; valid low cue does not establish coldness",
                f"negative_validation_scope={negative_validation_scope}",
            ),
        )
        thermal = replace(
            score.thermal,
            heat_source_likelihood=heat,
            source_type=CategoricalEstimate(
                OBJECT_PLUME_SOURCE,
                heat.confidence,
                OBJECT_PLUME_SOURCE,
                timestamp,
                heat.status,
                heat.evidence,
            )
            if heat.is_available
            else CategoricalEstimate.unavailable(
                source=OBJECT_PLUME_SOURCE, timestamp_seconds=timestamp
            ),
        )
        outputs[track] = ObjectPlumeEvidence(
            track,
            thermal,
            len(window),
            features.positive_frame_count,
            score.rejection_reasons,
            negative_validation_scope,
        )
        masks_by_track[track] = np.stack(masks)
    # A shared positive foreground cannot be confidently attributed to two
    # objects. Both remain unknown rather than assigning the plume arbitrarily.
    for index, track in enumerate(tracks):
        if track not in masks_by_track:
            continue
        first = masks_by_track[track]
        for other in tracks[index + 1 :]:
            if other not in masks_by_track:
                continue
            second = masks_by_track[other]
            intersection = np.count_nonzero(first & second)
            smaller = min(np.count_nonzero(first), np.count_nonzero(second))
            if smaller and intersection / smaller > 0.5:
                reason = (
                    "positive plume foreground is shared by multiple object regions",
                )
                outputs[track] = unknown(track, reason, len(items))
                outputs[other] = unknown(other, reason, len(items))
    return MappingProxyType(outputs)
