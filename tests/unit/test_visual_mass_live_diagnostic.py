import json

import pytest

from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.physical_attributes import (
    EstimateStatus,
    GeometryAttributes,
    MassAttributes,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.perception.physical_geometry import GEOMETRY_SOURCE
from robot.perception.visual_mass_live_diagnostic import (
    accepted_detection_index,
    evaluate_live_visual_mass_dataflow,
)
from robot.perception.visual_mass_runtime import VISUAL_MASS_RUNTIME_SOURCE


def estimate(value, unit, source, status=EstimateStatus.INFERRED):
    return ScalarEstimate(value, value * 0.9, value * 1.1, 0.7, unit, source, 1, status)


def attributes(
    *,
    geometry_source=GEOMETRY_SOURCE,
    mass_status=EstimateStatus.INFERRED,
):
    item = PhysicalAttributes.unknown(
        track_id=4,
        timestamp_seconds=1,
        semantic_class="chair",
        semantic_confidence=0.8,
        semantic_source="yolo11_segmentation_camera_lidar_fusion",
    )
    geometry = GeometryAttributes(
        estimate(0.5, "m", geometry_source),
        estimate(0.4, "m", geometry_source),
        estimate(0.8, "m", geometry_source),
        item.geometry.center_height_m,
        item.geometry.fall_height_m,
    )
    mass = MassAttributes(
        estimate(6.0, "kg", VISUAL_MASS_RUNTIME_SOURCE, mass_status)
    )
    return PhysicalAttributes(
        item.track_id,
        item.semantic_class,
        item.semantic_confidence,
        item.semantic_source,
        item.timestamp_seconds,
        mass,
        geometry,
        item.motion,
        item.stability,
        item.contact,
        item.thermal,
    )


def test_live_dataflow_passes_but_accuracy_claim_remains_blocked(tmp_path):
    report = evaluate_live_visual_mass_dataflow(
        (attributes(),), checkpoint_sha256="a" * 64
    )

    assert report.live_dataflow_valid
    assert not report.ready_for_sensor_domain_accuracy_claim
    assert report.tracks[0].dimensions_m == pytest.approx((0.5, 0.4, 0.8))
    assert report.tracks[0].geometry_statuses == ("inferred",) * 3
    output = tmp_path / "diagnostic.json"
    report.write_json(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["evidence_boundary"]["mass_reference"] == "unavailable"
    assert payload["evidence_boundary"]["fall_height_role"].startswith("not an input")


def test_authored_geometry_is_rejected_as_live_sensor_evidence():
    report = evaluate_live_visual_mass_dataflow(
        (attributes(geometry_source="isaac_authored_dimensions"),),
        checkpoint_sha256="b" * 64,
    )

    assert not report.live_dataflow_valid
    assert any("live LiDAR" in reason for reason in report.blockers)


def test_oracle_mass_is_rejected_as_runtime_inference():
    report = evaluate_live_visual_mass_dataflow(
        (attributes(mass_status=EstimateStatus.ORACLE),),
        checkpoint_sha256="c" * 64,
    )

    assert not report.live_dataflow_valid
    assert any("not marked inferred" in reason for reason in report.blockers)


def test_no_available_prediction_is_reported_explicitly():
    item = PhysicalAttributes.unknown(
        track_id=1, timestamp_seconds=1, semantic_class="unknown"
    )
    report = evaluate_live_visual_mass_dataflow((item,), checkpoint_sha256="d" * 64)

    assert report.tracks == ()
    assert report.blockers == (
        "no supported fused track produced a visual-mass estimate",
    )


def test_rejected_fusion_candidate_cannot_drive_mass_estimation():
    attempted = TrackSemanticAssociation(
        3,
        "insufficient_mask_overlap",
        10,
        2,
        1,
        0.1,
        0.05,
        False,
        (20.0, 30.0),
    )
    matched = TrackSemanticAssociation(
        4, "matched", 10, 7, 5, 0.5, 0.4, True, (20.0, 30.0)
    )

    assert accepted_detection_index(attempted) is None
    assert accepted_detection_index(matched) == 7
