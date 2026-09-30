"""Attach fixed-label visual specialist evidence to LiDAR track identities.

The camera--LiDAR fusion layer owns the image projection and track association.
This module only transfers aligned specialist results onto those existing track
IDs.  Missing, stale, or ambiguous evidence stays unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from types import MappingProxyType
from typing import Mapping

from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.contact_cues import (
    DangerousPartSegmentation,
    estimate_contact_cues,
)
from robot.perception.hot_soup_detector import (
    HotSoupDetectionCueResult,
    HotSoupDetectionFrame,
)
from robot.perception.physical_attributes import (
    ContactAttributes,
    PhysicalAttributes,
    ThermalAttributes,
)
from robot.perception.semantic_detection import PixelBoundingBox


def _finite_nonnegative(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class SpecialistAssociationFrame:
    """Track projection evidence tied to one camera frame."""

    source_frame_id: int
    source_timestamp_seconds: float
    image_height: int
    image_width: int
    associations: tuple[TrackSemanticAssociation, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_frame_id, bool)
            or not isinstance(self.source_frame_id, int)
            or self.source_frame_id < 0
        ):
            raise ValueError("source_frame_id must be a nonnegative integer")
        timestamp = _finite_nonnegative(
            self.source_timestamp_seconds,
            name="source_timestamp_seconds",
        )
        if (
            isinstance(self.image_height, bool)
            or not isinstance(self.image_height, int)
            or isinstance(self.image_width, bool)
            or not isinstance(self.image_width, int)
            or self.image_height <= 0
            or self.image_width <= 0
        ):
            raise ValueError("image dimensions must be positive")
        associations = tuple(self.associations)
        if any(
            not isinstance(item, TrackSemanticAssociation)
            for item in associations
        ):
            raise TypeError("associations must contain TrackSemanticAssociation values")
        track_ids = [item.track_id for item in associations]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("associations contains duplicate track IDs")
        object.__setattr__(self, "source_timestamp_seconds", timestamp)
        object.__setattr__(self, "associations", associations)


@dataclass(frozen=True)
class SpecialistTrackDiagnostics:
    """Auditable association outcome for one specialist update."""

    contact_track_ids: tuple[int, ...] = ()
    thermal_track_ids: tuple[int, ...] = ()
    ambiguous_thermal_track_ids: tuple[int, ...] = ()
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrackSpecialistEvidence:
    """Freshness-gated per-track contact and thermal provider."""

    contact_by_track_id: Mapping[int, ContactAttributes] = field(
        default_factory=dict
    )
    thermal_by_track_id: Mapping[int, ThermalAttributes] = field(
        default_factory=dict
    )
    contact_timestamp_seconds: float | None = None
    thermal_timestamp_seconds: float | None = None
    maximum_age_seconds: float = 0.5
    diagnostics: SpecialistTrackDiagnostics = field(
        default_factory=SpecialistTrackDiagnostics
    )

    def __post_init__(self) -> None:
        maximum_age = _finite_nonnegative(
            self.maximum_age_seconds,
            name="maximum_age_seconds",
        )
        contact = dict(self.contact_by_track_id)
        thermal = dict(self.thermal_by_track_id)
        for track_id, value in contact.items():
            if not isinstance(track_id, int) or isinstance(track_id, bool) or track_id < 0:
                raise ValueError("contact track IDs must be nonnegative integers")
            if not isinstance(value, ContactAttributes):
                raise TypeError("contact evidence must contain ContactAttributes")
        for track_id, value in thermal.items():
            if not isinstance(track_id, int) or isinstance(track_id, bool) or track_id < 0:
                raise ValueError("thermal track IDs must be nonnegative integers")
            if not isinstance(value, ThermalAttributes):
                raise TypeError("thermal evidence must contain ThermalAttributes")
        for name in ("contact_timestamp_seconds", "thermal_timestamp_seconds"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite_nonnegative(value, name=name),
                )
        if contact and self.contact_timestamp_seconds is None:
            raise ValueError("contact evidence requires a source timestamp")
        if thermal and self.thermal_timestamp_seconds is None:
            raise ValueError("thermal evidence requires a source timestamp")
        if not isinstance(self.diagnostics, SpecialistTrackDiagnostics):
            raise TypeError("diagnostics must be SpecialistTrackDiagnostics")
        object.__setattr__(self, "contact_by_track_id", MappingProxyType(contact))
        object.__setattr__(self, "thermal_by_track_id", MappingProxyType(thermal))
        object.__setattr__(self, "maximum_age_seconds", maximum_age)

    def contact_for(self, track_id: int, timestamp_seconds: float) -> ContactAttributes:
        """Return fresh contact evidence, otherwise explicit unknowns."""

        timestamp = _finite_nonnegative(timestamp_seconds, name="timestamp_seconds")
        value = self.contact_by_track_id.get(track_id)
        if value is not None and self._is_fresh(
            self.contact_timestamp_seconds,
            timestamp,
        ):
            return value
        return PhysicalAttributes.unknown(
            track_id=track_id,
            timestamp_seconds=timestamp,
        ).contact

    def thermal_for(self, track_id: int, timestamp_seconds: float) -> ThermalAttributes:
        """Return fresh thermal evidence, otherwise explicit unknowns."""

        timestamp = _finite_nonnegative(timestamp_seconds, name="timestamp_seconds")
        value = self.thermal_by_track_id.get(track_id)
        if value is not None and self._is_fresh(
            self.thermal_timestamp_seconds,
            timestamp,
        ):
            return value
        return PhysicalAttributes.unknown(
            track_id=track_id,
            timestamp_seconds=timestamp,
        ).thermal

    def _is_fresh(self, source_timestamp: float | None, timestamp: float) -> bool:
        if source_timestamp is None or source_timestamp > timestamp:
            return False
        return timestamp - source_timestamp <= self.maximum_age_seconds


def build_track_specialist_evidence(
    association_frame: SpecialistAssociationFrame,
    *,
    contact_segmentations: Mapping[int, DangerousPartSegmentation] | None = None,
    thermal_attributes_by_detection_index: Mapping[int, ThermalAttributes]
    | None = None,
    hot_soup_frame: HotSoupDetectionFrame | None = None,
    hot_soup_result: HotSoupDetectionCueResult | None = None,
    maximum_age_seconds: float = 0.5,
) -> TrackSpecialistEvidence:
    """Build provider-ready evidence from aligned fixed-label results.

    Contact segmentation and continuous thermal keys are detection indices
    from the association frame. Hot-soup evidence is assigned through the selected container box and the
    projected median of each visible LiDAR track.  A box containing multiple
    track medians is ambiguous and is not assigned.
    """

    if not isinstance(association_frame, SpecialistAssociationFrame):
        raise TypeError("association_frame must be SpecialistAssociationFrame")
    contact_inputs = dict(contact_segmentations or {})
    for index, segmentation in contact_inputs.items():
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("contact segmentation indices must be nonnegative integers")
        if not isinstance(segmentation, DangerousPartSegmentation):
            raise TypeError(
                "contact_segmentations must contain DangerousPartSegmentation values"
            )
        if not math.isclose(
            segmentation.timestamp_seconds,
            association_frame.source_timestamp_seconds,
            abs_tol=1e-9,
        ):
            raise ValueError("contact segmentation timestamp must match association frame")
    contact: dict[int, ContactAttributes] = {}
    reasons: list[str] = []
    for association in association_frame.associations:
        index = association.detection_index
        if not association.is_matched or index is None or index not in contact_inputs:
            continue
        segmentation = contact_inputs[index]
        contact[association.track_id] = estimate_contact_cues(
            segmentation,
            timestamp_seconds=association_frame.source_timestamp_seconds,
        )

    thermal: dict[int, ThermalAttributes] = {}
    ambiguous_thermal: tuple[int, ...] = ()
    thermal_timestamp: float | None = None
    continuous_thermal = dict(thermal_attributes_by_detection_index or {})
    if continuous_thermal and (
        hot_soup_frame is not None or hot_soup_result is not None
    ):
        raise ValueError(
            "continuous thermal attributes and legacy hot-soup evidence are "
            "mutually exclusive"
        )
    for index, attributes in continuous_thermal.items():
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("thermal detection indices must be nonnegative integers")
        if not isinstance(attributes, ThermalAttributes):
            raise TypeError(
                "thermal_attributes_by_detection_index must contain ThermalAttributes"
            )
        _validate_thermal_timestamp(
            attributes,
            association_frame.source_timestamp_seconds,
        )
        if not attributes.heat_source_likelihood.is_available:
            reasons.append(
                f"thermal detection index {index} has unavailable heat-source evidence"
            )
            continue
        candidates = sorted(
            association.track_id
            for association in association_frame.associations
            if association.is_matched and association.detection_index == index
        )
        if len(candidates) == 1:
            thermal[candidates[0]] = attributes
        elif len(candidates) > 1:
            ambiguous_thermal = tuple(
                sorted(set(ambiguous_thermal).union(candidates))
            )
            reasons.append(
                f"thermal detection index {index} maps to multiple LiDAR tracks"
            )
        else:
            reasons.append(
                f"thermal detection index {index} has no accepted LiDAR track"
            )
    if continuous_thermal:
        thermal_timestamp = association_frame.source_timestamp_seconds
    if (hot_soup_frame is None) != (hot_soup_result is None):
        raise ValueError("hot_soup_frame and hot_soup_result must be supplied together")
    if hot_soup_frame is not None and hot_soup_result is not None:
        if not isinstance(hot_soup_result, HotSoupDetectionCueResult):
            raise TypeError("hot_soup_result must be HotSoupDetectionCueResult")
        _validate_aligned_hot_soup_frame(association_frame, hot_soup_frame)
        thermal_timestamp = hot_soup_frame.source_timestamp_seconds
        selected = hot_soup_result.selected_container_index
        if selected is None:
            reasons.append("hot-soup result has no selected container")
        else:
            if selected < 0 or selected >= len(hot_soup_frame.detections):
                raise ValueError("selected hot-soup container index is out of range")
            selected_detection = hot_soup_frame.detections[selected]
            if selected_detection.semantic_class != "open_container":
                raise ValueError("selected hot-soup detection must be an open container")
            box = selected_detection.bounding_box
            candidates = _tracks_inside_box(association_frame.associations, box)
            if len(candidates) == 1:
                thermal[candidates[0]] = hot_soup_result.cue_result.thermal
            elif candidates:
                ambiguous_thermal = tuple(candidates)
                reasons.append("hot-soup container contains multiple projected tracks")
            else:
                reasons.append("hot-soup container contains no projected track")

    return TrackSpecialistEvidence(
        contact_by_track_id=contact,
        thermal_by_track_id=thermal,
        contact_timestamp_seconds=(
            association_frame.source_timestamp_seconds if contact_inputs else None
        ),
        thermal_timestamp_seconds=thermal_timestamp,
        maximum_age_seconds=maximum_age_seconds,
        diagnostics=SpecialistTrackDiagnostics(
            contact_track_ids=tuple(sorted(contact)),
            thermal_track_ids=tuple(sorted(thermal)),
            ambiguous_thermal_track_ids=ambiguous_thermal,
            rejection_reasons=tuple(reasons),
        ),
    )


def _validate_aligned_hot_soup_frame(
    association_frame: SpecialistAssociationFrame,
    hot_soup_frame: HotSoupDetectionFrame,
) -> None:
    if not isinstance(hot_soup_frame, HotSoupDetectionFrame):
        raise TypeError("hot_soup_frame must be HotSoupDetectionFrame")
    if hot_soup_frame.source_frame_id != association_frame.source_frame_id:
        raise ValueError("hot-soup and association frame IDs must match")
    if not math.isclose(
        hot_soup_frame.source_timestamp_seconds,
        association_frame.source_timestamp_seconds,
        abs_tol=1e-9,
    ):
        raise ValueError("hot-soup and association timestamps must match")
    if (
        hot_soup_frame.image_height != association_frame.image_height
        or hot_soup_frame.image_width != association_frame.image_width
    ):
        raise ValueError("hot-soup and association image dimensions must match")


def _validate_thermal_timestamp(
    thermal: ThermalAttributes,
    expected_timestamp: float,
) -> None:
    for item in fields(thermal):
        estimate = getattr(thermal, item.name)
        if not math.isclose(
            estimate.timestamp_seconds,
            expected_timestamp,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "continuous thermal timestamp must match association frame"
            )


def _tracks_inside_box(
    associations: tuple[TrackSemanticAssociation, ...],
    box: PixelBoundingBox,
) -> list[int]:
    candidates = []
    for association in associations:
        point = association.projected_median_xy
        if point is None:
            continue
        x, y = point
        if box.x_min <= x <= box.x_max and box.y_min <= y <= box.y_max:
            candidates.append(association.track_id)
    return sorted(candidates)
