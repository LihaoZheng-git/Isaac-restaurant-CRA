"""Auditable result for one live fixed-label contact-specialist invocation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from robot.perception.contact_detection_adapter import ContactPartAssociationResult
from robot.perception.semantic_detection import SemanticDetectionFrame
from robot.perception.specialist_track_evidence import TrackSpecialistEvidence


@dataclass(frozen=True)
class LiveContactDiagnostic:
    checkpoint_sha256: str
    model_name: str
    part_detection_count: int
    accepted_part_count: int
    contact_track_ids: tuple[int, ...]
    evidence_status: str
    pipeline_valid: bool
    limitations: tuple[str, ...] = (
        "only exposed-edge geometry has nonzero held-out baseline performance",
        "sharpness, hardness, and route orientation remain unavailable",
        "detector non-response is not evidence of zero contact hazard",
    )

    @property
    def ready_for_complete_contact_factor(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ready_for_complete_contact_factor"] = (
            self.ready_for_complete_contact_factor
        )
        return payload

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def evaluate_live_contact_dataflow(
    part_frame: SemanticDetectionFrame,
    association: ContactPartAssociationResult,
    specialist: TrackSpecialistEvidence,
    *,
    checkpoint_sha256: str,
) -> LiveContactDiagnostic:
    """Summarize invocation separately from positive contact evidence."""

    digest = str(checkpoint_sha256).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("checkpoint_sha256 must be a hexadecimal SHA-256 digest")
    if not isinstance(part_frame, SemanticDetectionFrame):
        raise TypeError("part_frame must be a SemanticDetectionFrame")
    if not isinstance(association, ContactPartAssociationResult):
        raise TypeError("association must be a ContactPartAssociationResult")
    if not isinstance(specialist, TrackSpecialistEvidence):
        raise TypeError("specialist must be TrackSpecialistEvidence")
    diagnostics = association.diagnostics
    tracks = specialist.diagnostics.contact_track_ids
    pipeline_valid = (
        diagnostics.part_detection_count == len(part_frame.detections)
        and diagnostics.accepted_part_count >= len(tracks)
    )
    return LiveContactDiagnostic(
        checkpoint_sha256=digest,
        model_name=part_frame.model_name,
        part_detection_count=diagnostics.part_detection_count,
        accepted_part_count=diagnostics.accepted_part_count,
        contact_track_ids=tracks,
        evidence_status="inferred" if tracks else "unavailable",
        pipeline_valid=pipeline_valid,
    )
