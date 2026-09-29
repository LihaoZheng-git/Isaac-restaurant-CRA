"""Auditable status for one stationary hot-soup specialist invocation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from robot.perception.hot_soup_detector import (
    HotSoupDetectionCueResult,
    HotSoupDetectionFrame,
)
from robot.perception.specialist_track_evidence import TrackSpecialistEvidence


@dataclass(frozen=True)
class LiveThermalDiagnostic:
    checkpoint_sha256: str
    model_name: str
    plume_detection_count: int
    container_detection_count: int
    candidate_pair_count: int
    spatial_association_score: float | None
    thermal_track_ids: tuple[int, ...]
    ambiguous_thermal_track_ids: tuple[int, ...]
    evidence_status: str
    pipeline_valid: bool
    limitations: tuple[str, ...] = (
        "visual cues do not measure temperature or prove steam",
        "detector non-response is not evidence of zero thermal hazard",
        "carried state, spill state, and route interaction are separate inputs",
        "sensor-domain accuracy requires held-out project-owned real evaluation",
    )

    @property
    def ready_for_sensor_domain_accuracy_claim(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ready_for_sensor_domain_accuracy_claim"] = (
            self.ready_for_sensor_domain_accuracy_claim
        )
        return payload

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def evaluate_live_thermal_dataflow(
    frame: HotSoupDetectionFrame,
    result: HotSoupDetectionCueResult,
    specialist: TrackSpecialistEvidence,
    *,
    checkpoint_sha256: str,
) -> LiveThermalDiagnostic:
    """Summarize model invocation separately from positive thermal evidence."""

    digest = str(checkpoint_sha256).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("checkpoint_sha256 must be a hexadecimal SHA-256 digest")
    if not isinstance(frame, HotSoupDetectionFrame):
        raise TypeError("frame must be HotSoupDetectionFrame")
    if not isinstance(result, HotSoupDetectionCueResult):
        raise TypeError("result must be HotSoupDetectionCueResult")
    if not isinstance(specialist, TrackSpecialistEvidence):
        raise TypeError("specialist must be TrackSpecialistEvidence")
    plume_count = sum(
        item.semantic_class == "visible_hot_soup_plume" for item in frame.detections
    )
    container_count = sum(
        item.semantic_class == "open_container" for item in frame.detections
    )
    tracks = specialist.diagnostics.thermal_track_ids
    ambiguous = specialist.diagnostics.ambiguous_thermal_track_ids
    heat_available = result.cue_result.thermal.heat_source_likelihood.is_available
    pipeline_valid = (
        result.candidate_pair_count <= plume_count * container_count
        and not (tracks and not heat_available)
        and not (tracks and ambiguous)
    )
    return LiveThermalDiagnostic(
        checkpoint_sha256=digest,
        model_name=frame.model_name,
        plume_detection_count=plume_count,
        container_detection_count=container_count,
        candidate_pair_count=result.candidate_pair_count,
        spatial_association_score=result.spatial_association_score,
        thermal_track_ids=tracks,
        ambiguous_thermal_track_ids=ambiguous,
        evidence_status="inferred" if tracks and heat_available else "unavailable",
        pipeline_valid=pipeline_valid,
    )
