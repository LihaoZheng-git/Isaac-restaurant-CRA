from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.hot_soup_detector import (
    HotSoupDetection,
    HotSoupDetectionFrame,
    estimate_hot_soup_from_detections,
)
from robot.perception.semantic_detection import PixelBoundingBox
from robot.perception.specialist_track_evidence import (
    SpecialistAssociationFrame,
    build_track_specialist_evidence,
)
from robot.perception.thermal_live_diagnostic import (
    evaluate_live_thermal_dataflow,
)


def frame(*detections):
    return HotSoupDetectionFrame(7, 2.0, 100, 100, detections, "fixture.pt", 0.01)


def association():
    return TrackSemanticAssociation(
        4,
        "matched",
        4,
        0,
        4,
        1.0,
        1.0,
        True,
        (50.0, 65.0),
    )


def positive_frame():
    return frame(
        HotSoupDetection(
            0,
            "visible_hot_soup_plume",
            0.9,
            PixelBoundingBox(40, 10, 60, 45),
        ),
        HotSoupDetection(
            1,
            "open_container",
            0.8,
            PixelBoundingBox(30, 45, 70, 80),
        ),
    )


def specialist(item, result):
    return build_track_specialist_evidence(
        SpecialistAssociationFrame(7, 2.0, 100, 100, (association(),)),
        hot_soup_frame=item,
        hot_soup_result=result,
    )


def test_positive_stationary_dataflow_reports_inferred_track_without_accuracy_claim():
    item = positive_frame()
    result = estimate_hot_soup_from_detections(item, timestamp_seconds=2.0)
    diagnostic = evaluate_live_thermal_dataflow(
        item,
        result,
        specialist(item, result),
        checkpoint_sha256="a" * 64,
    )
    assert diagnostic.plume_detection_count == 1
    assert diagnostic.container_detection_count == 1
    assert diagnostic.candidate_pair_count == 1
    assert diagnostic.thermal_track_ids == (4,)
    assert diagnostic.evidence_status == "inferred"
    assert diagnostic.pipeline_valid
    assert not diagnostic.ready_for_sensor_domain_accuracy_claim


def test_empty_detector_result_is_valid_invocation_but_unavailable_evidence():
    item = frame()
    result = estimate_hot_soup_from_detections(item, timestamp_seconds=2.0)
    diagnostic = evaluate_live_thermal_dataflow(
        item,
        result,
        specialist(item, result),
        checkpoint_sha256="b" * 64,
    )
    assert diagnostic.plume_detection_count == 0
    assert diagnostic.thermal_track_ids == ()
    assert diagnostic.evidence_status == "unavailable"
    assert diagnostic.pipeline_valid
