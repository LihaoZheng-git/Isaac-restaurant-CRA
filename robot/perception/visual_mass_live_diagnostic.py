"""Provenance checks for stationary live visual-mass diagnostics.

Passing this diagnostic establishes a live RGB-plus-LiDAR dataflow check. It
does not establish mass accuracy without independent physical mass labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.physical_attributes import EstimateStatus, PhysicalAttributes
from robot.perception.physical_geometry import GEOMETRY_SOURCE
from robot.perception.sam_instance_segmentation import SAM_INSTANCE_SOURCE
from robot.perception.visual_mass_runtime import (
    VISUAL_MASS_RUNTIME_SOURCE,
    CATEGORY_FREE_MASS_EVIDENCE,
)


def accepted_detection_index(
    association: TrackSemanticAssociation | None,
) -> int | None:
    """Return a detection only after the camera--LiDAR gate accepted it."""

    if association is None or not association.is_matched:
        return None
    return association.detection_index


@dataclass(frozen=True)
class LiveVisualMassTrackDiagnostic:
    """One live fused track and the provenance used for mass inference."""

    track_id: int
    semantic_class: str
    semantic_source: str
    geometry_statuses: tuple[str, str, str]
    geometry_sources: tuple[str, str, str]
    dimensions_m: tuple[float, float, float]
    mass_status: str
    mass_source: str
    predicted_mass_kg: float
    lower_bound_kg: float
    upper_bound_kg: float
    confidence: float


@dataclass(frozen=True)
class LiveVisualMassDiagnostic:
    """Result of checking a stationary live visual-mass dataflow."""

    checkpoint_sha256: str
    tracks: tuple[LiveVisualMassTrackDiagnostic, ...]
    blockers: tuple[str, ...]
    accuracy_claim_blockers: tuple[str, ...] = (
        "no independent measured mass labels for the live restaurant objects",
        "the ABO checkpoint is not calibrated on live Isaac restaurant imagery",
    )

    @property
    def live_dataflow_valid(self) -> bool:
        return not self.blockers

    @property
    def ready_for_sensor_domain_accuracy_claim(self) -> bool:
        return not self.blockers and not self.accuracy_claim_blockers

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["live_dataflow_valid"] = self.live_dataflow_valid
        payload["ready_for_sensor_domain_accuracy_claim"] = (
            self.ready_for_sensor_domain_accuracy_claim
        )
        payload["evidence_boundary"] = {
            "rgb_segmentation": "inferred from the configured instance model; source recorded per track",
            "geometry": "inferred from current LiDAR track returns",
            "mass": "inferred by the configured ABO RGB-plus-geometry checkpoint",
            "mass_reference": "unavailable",
            "fall_height_role": "not an input to visual mass estimation",
        }
        return payload

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def evaluate_live_visual_mass_dataflow(
    attributes: Iterable[PhysicalAttributes],
    *,
    checkpoint_sha256: str,
    instance_associations: Iterable[TrackSemanticAssociation] = (),
) -> LiveVisualMassDiagnostic:
    """Check that available masses use inferred RGB and live LiDAR geometry."""

    digest = str(checkpoint_sha256).strip().lower()
    blockers: list[str] = []
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("checkpoint_sha256 must be a hexadecimal SHA-256 digest")

    track_rows = []
    matched_instance_ids = {
        item.track_id for item in instance_associations if item.is_matched
    }
    for item in attributes:
        mass = item.mass.mass_kg
        if not mass.is_available:
            continue
        dimensions = (
            item.geometry.width_m,
            item.geometry.depth_m,
            item.geometry.height_m,
        )
        prefix = f"track {item.track_id}"
        if any(not estimate.is_available for estimate in dimensions):
            blockers.append(f"{prefix}: a visual-mass dimension is unavailable")
            continue
        if any(
            estimate.status not in {EstimateStatus.MEASURED, EstimateStatus.INFERRED}
            for estimate in dimensions
        ):
            blockers.append(f"{prefix}: geometry contains non-sensor provenance")
        if any(estimate.source != GEOMETRY_SOURCE for estimate in dimensions):
            blockers.append(f"{prefix}: geometry source is not the live LiDAR adapter")
        if mass.status is not EstimateStatus.INFERRED:
            blockers.append(f"{prefix}: mass is not marked inferred")
        if mass.source != VISUAL_MASS_RUNTIME_SOURCE:
            blockers.append(
                f"{prefix}: mass source is not the visual-mass runtime adapter"
            )
        if item.semantic_class == "unknown":
            if not (
                item.track_id in matched_instance_ids
                and item.semantic_source == SAM_INSTANCE_SOURCE
                and CATEGORY_FREE_MASS_EVIDENCE in mass.evidence
                and f"accepted_instance_track_id={item.track_id}" in mass.evidence
            ):
                blockers.append(
                    f"{prefix}: category-free instance association is unavailable"
                )
        track_rows.append(
            LiveVisualMassTrackDiagnostic(
                track_id=item.track_id,
                semantic_class=item.semantic_class,
                semantic_source=item.semantic_source,
                geometry_statuses=tuple(
                    estimate.status.value for estimate in dimensions
                ),
                geometry_sources=tuple(estimate.source for estimate in dimensions),
                dimensions_m=tuple(float(estimate.value) for estimate in dimensions),
                mass_status=mass.status.value,
                mass_source=mass.source,
                predicted_mass_kg=float(mass.value),
                lower_bound_kg=float(mass.lower_bound),
                upper_bound_kg=float(mass.upper_bound),
                confidence=mass.confidence,
            )
        )
    if not track_rows:
        blockers.append("no supported fused track produced a visual-mass estimate")
    return LiveVisualMassDiagnostic(digest, tuple(track_rows), tuple(blockers))
