"""Reports for positive fixed-label contact sensor-replay diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from robot.perception.contact_detection_adapter import ContactPartAssociationResult
from robot.perception.semantic_detection import SemanticDetectionFrame


@dataclass(frozen=True)
class ContactReplayDiagnostic:
    image_sha256: str
    object_checkpoint_sha256: str
    contact_checkpoint_sha256: str
    object_detection_count: int
    part_detection_count: int
    accepted_part_count: int
    accepted_parent_classes: tuple[str, ...]
    ambiguous_part_count: int
    unmatched_part_count: int
    positive_association_valid: bool
    evidence_scope: str = "held_out_paco_image_replay"
    limitations: tuple[str, ...] = (
        "not a live Isaac camera or LiDAR observation",
        "PACO exposed-edge geometry is not physical sharpness or hardness",
        "does not validate pointed tips, nails, protrusions, or jagged surfaces",
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def evaluate_contact_replay(
    object_frame: SemanticDetectionFrame,
    part_frame: SemanticDetectionFrame,
    association: ContactPartAssociationResult,
    *,
    image_sha256: str,
    object_checkpoint_sha256: str,
    contact_checkpoint_sha256: str,
) -> ContactReplayDiagnostic:
    """Require at least one accepted part-to-parent association."""

    if not isinstance(object_frame, SemanticDetectionFrame):
        raise TypeError("object_frame must be a SemanticDetectionFrame")
    if not isinstance(part_frame, SemanticDetectionFrame):
        raise TypeError("part_frame must be a SemanticDetectionFrame")
    if not isinstance(association, ContactPartAssociationResult):
        raise TypeError("association must be a ContactPartAssociationResult")
    digests = (image_sha256, object_checkpoint_sha256, contact_checkpoint_sha256)
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in digests
    ):
        raise ValueError("all hashes must be lowercase hexadecimal SHA-256 digests")
    indices = tuple(sorted(association.segmentations_by_object_detection_index))
    if any(index >= len(object_frame.detections) for index in indices):
        raise ValueError("association references an absent parent detection")
    parent_classes = tuple(
        object_frame.detections[index].semantic_class for index in indices
    )
    diagnostics = association.diagnostics
    return ContactReplayDiagnostic(
        image_sha256=image_sha256,
        object_checkpoint_sha256=object_checkpoint_sha256,
        contact_checkpoint_sha256=contact_checkpoint_sha256,
        object_detection_count=len(object_frame.detections),
        part_detection_count=len(part_frame.detections),
        accepted_part_count=diagnostics.accepted_part_count,
        accepted_parent_classes=parent_classes,
        ambiguous_part_count=len(diagnostics.ambiguous_part_indices),
        unmatched_part_count=len(diagnostics.unmatched_part_indices),
        positive_association_valid=bool(indices and diagnostics.accepted_part_count),
    )
