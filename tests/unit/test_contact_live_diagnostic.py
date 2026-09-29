import numpy as np

from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.contact_detection_adapter import (
    associate_contact_parts_to_objects,
)
from robot.perception.contact_live_diagnostic import evaluate_live_contact_dataflow
from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetection,
    SemanticDetectionFrame,
)
from robot.perception.specialist_track_evidence import (
    SpecialistAssociationFrame,
    build_track_specialist_evidence,
)


def frame(name, mask=None, *, model="contact.pt"):
    detections = ()
    if mask is not None:
        detections = (
            SemanticDetection(0, name, 0.8, PixelBoundingBox(2, 2, 8, 8), mask),
        )
    return SemanticDetectionFrame(3, 2.0, 10, 10, detections, model, 0.01)


def association():
    return TrackSemanticAssociation(4, "matched", 5, 0, 5, 1.0, 1.0, True, (5, 5))


def test_live_contact_diagnostic_separates_invocation_from_complete_factor():
    parent_mask = np.zeros((10, 10), dtype=bool)
    parent_mask[1:9, 1:9] = True
    part_mask = np.zeros((10, 10), dtype=bool)
    part_mask[2:4, 2:8] = True
    objects = frame("knife", parent_mask, model="objects.pt")
    parts = frame("exposed_edge", part_mask)
    result = associate_contact_parts_to_objects(objects, parts)
    specialist = build_track_specialist_evidence(
        SpecialistAssociationFrame(3, 2.0, 10, 10, (association(),)),
        contact_segmentations=result.segmentations_by_object_detection_index,
    )

    report = evaluate_live_contact_dataflow(
        parts, result, specialist, checkpoint_sha256="a" * 64
    )
    assert report.pipeline_valid
    assert report.evidence_status == "inferred"
    assert report.contact_track_ids == (4,)
    assert not report.ready_for_complete_contact_factor


def test_no_detection_remains_unavailable_without_failing_invocation():
    empty = frame("exposed_edge")
    objects = frame("chair")
    result = associate_contact_parts_to_objects(objects, empty)
    specialist = build_track_specialist_evidence(
        SpecialistAssociationFrame(3, 2.0, 10, 10, (association(),)),
        contact_segmentations=result.segmentations_by_object_detection_index,
    )

    report = evaluate_live_contact_dataflow(
        empty, result, specialist, checkpoint_sha256="b" * 64
    )
    assert report.pipeline_valid
    assert report.evidence_status == "unavailable"
    assert report.contact_track_ids == ()
