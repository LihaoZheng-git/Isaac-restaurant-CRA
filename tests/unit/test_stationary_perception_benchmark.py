from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from robot.evaluation.restaurant_evidence_records import (
    EvaluationArtifactFingerprint,
    EvaluationArtifactKind,
    EvaluationRunIdentity,
)
from robot.evaluation.stationary_perception_benchmark import (
    StationaryCaptureChecks,
    summarize_stationary_captures,
    write_stationary_sensor_capture,
)
from robot.perception.physical_attributes import (
    EstimateStatus,
    MassAttributes,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.risk.physical_factor_adapter import adapt_physical_factors


def _checks(**changes):
    return replace(StationaryCaptureChecks(
        sensor_pipeline_passed=True,
        robot_position_drift_m=0.001,
        robot_yaw_drift_degrees=0.001,
        maximum_position_drift_m=0.005,
        maximum_yaw_drift_degrees=0.1,
        camera_lidar_age_seconds=0.02,
        maximum_sensor_age_seconds=0.15,
        published_motion_count=0,
        motion_sample_count=10,
        matched_camera_lidar_track_count=1,
        nonzero_wheel_command_count=0,
    ), **changes)


def _attributes(status=EstimateStatus.INFERRED, semantic_class="chair"):
    base = PhysicalAttributes.unknown(
        track_id=7, timestamp_seconds=2.0,
        semantic_class=semantic_class, semantic_confidence=0.8,
        semantic_source="camera_lidar_mask_association",
    )

    def scalar(value, unit):
        return ScalarEstimate(value, value / 2, value * 2, 0.25, unit,
                              "live_sensor_estimator", 2.0, status)

    return replace(
        base,
        mass=MassAttributes(scalar(10.0, "kg")),
        geometry=replace(
            base.geometry, width_m=scalar(0.4, "m"),
            depth_m=scalar(0.3, "m"), height_m=scalar(0.9, "m"),
        ),
    )


def _capture(tmp_path, capture_id="001", attributes=None, checks=None):
    item = _attributes() if attributes is None else attributes
    fingerprint = EvaluationArtifactFingerprint(
        name="model", kind=EvaluationArtifactKind.MODEL_WEIGHTS,
        sha256="a" * 64,
    )
    return write_stationary_sensor_capture(
        tmp_path / capture_id,
        identity=EvaluationRunIdentity(
            run_id="stationary-001", capture_id=capture_id,
            code_revision="0123456789abcdef", artifacts=(fingerprint,),
        ),
        attributes=(item,), adapted=(adapt_physical_factors(item, timestamp_seconds=2.0),),
        checks=_checks() if checks is None else checks,
        configuration={"fixed_lambda": 1.25},
        model_invocations={"contact": {"invoked": True, "pipeline_valid": True}},
        artifact_paths={"raw_rgb": "local.png"},
        oracle_scoring={"mass_reference": None, "full_height_m": 1.0},
    )


@pytest.mark.parametrize("changes,reason", [
    ({"robot_position_drift_m": 0.006}, "position drift"),
    ({"camera_lidar_age_seconds": 0.3}, "sensor age"),
    ({"nonzero_wheel_command_count": 1}, "wheel command"),
    ({"published_motion_count": 1}, "nonzero object motion"),
    ({"matched_camera_lidar_track_count": 0}, "association"),
    ({"motion_sample_count": 0}, "samples unavailable"),
])
def test_stationary_acceptance_rejects_invalid_dataflow(changes, reason):
    assert any(reason in item for item in _checks(**changes).blockers)


@pytest.mark.parametrize("changes", [
    {"robot_position_drift_m": float("nan")},
    {"maximum_sensor_age_seconds": -1},
    {"motion_sample_count": True},
    {"sensor_pipeline_passed": 1},
])
def test_check_contract_rejects_invalid_values(changes):
    with pytest.raises((ValueError, TypeError)):
        _checks(**changes)


def test_capture_preserves_numeric_evidence_and_separate_unknowns(tmp_path):
    payload = _capture(tmp_path)
    saved = json.loads((tmp_path / "001" / "capture.json").read_text())
    track = saved["tracks"][0]
    mass = track["physical_attributes"]["mass"]["mass_kg"]
    assert mass["value"] == 10
    assert mass["lower_bound"] == 5
    assert mass["confidence"] == 0.25
    assert mass["status"] == "inferred"
    fall = track["physical_attributes"]["geometry"]["fall_height_m"]
    assert fall["value"] is None and fall["status"] == "unavailable"
    assert track["physical_attributes"]["geometry"]["height_m"]["value"] == 0.9
    factors = track["factor_diagnostics"]
    assert not factors["factor_specific"]["thermal"]["available"]
    assert not factors["factor_specific"]["contact"]["available"]
    assert factors["aggregate"]["conservative_phi"] == 1.0
    assert payload["oracle_scoring"]["available_to_runtime"] is False
    assert len(payload["scenario_record_paths"]) == 2
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in payload["scenario_record_paths"]]
    assert all(not r["runtime_evidence"]["readiness"]["aggregate_phi_reportable"] for r in records)


@pytest.mark.parametrize("status", [EstimateStatus.PRIOR, EstimateStatus.ORACLE, EstimateStatus.FALLBACK])
def test_prior_or_oracle_cannot_count_as_sensor_mass_or_geometry(tmp_path, status):
    payload = _capture(tmp_path, attributes=_attributes(status))
    assert not payload["tracks"][0]["sensor_geometry_available"]
    assert not payload["tracks"][0]["sensor_mass_available"]
    assert payload["tracks"][0]["physical_attributes"]["mass"]["mass_kg"]["status"] == status.value


def test_camera_class_is_not_relabelled_to_create_scenario_records(tmp_path):
    payload = _capture(tmp_path, attributes=_attributes(semantic_class="bowl"))
    assert payload["scenario_record_paths"] == []
    assert payload["tracks"][0]["semantic_class"] == "bowl"


def test_dataflow_does_not_claim_factor_accuracy_or_motion_readiness(tmp_path):
    captures = tuple(_capture(tmp_path, str(index)) for index in range(3))
    summary = summarize_stationary_captures(captures)
    assert summary["dataflow_passed"]
    assert summary["validated_capture_counts"] == {"geometry": 3, "visual_mass": 3, "contact_invocation": 3}
    assert not summary["ready_for_sensor_motion"]
    assert len(summary["factor_accuracy_blockers"]) == 5
    with pytest.raises(ValueError, match="unique"):
        summarize_stationary_captures((captures[0], captures[0]))
    with pytest.raises(FileExistsError):
        _capture(tmp_path, "0")


def test_summary_rechecks_stationarity_and_requires_matching_fingerprints(tmp_path):
    captures = tuple(_capture(tmp_path, str(index)) for index in range(3))
    altered = deepcopy(captures[0])
    altered["checks"]["nonzero_wheel_command_count"] = 1
    altered["artifacts"][0]["sha256"] = "b" * 64
    summary = summarize_stationary_captures((altered, *captures[1:]))
    assert not summary["dataflow_passed"]
    assert len(summary["dataflow_blockers"]) == 2
    altered["evidence_mode"] = "oracle"
    with pytest.raises(ValueError, match="oracle/prior"):
        summarize_stationary_captures((altered,))


def test_capture_requires_consistent_attributes_and_adapter_identity(tmp_path):
    item = _attributes()
    adapted = adapt_physical_factors(item, timestamp_seconds=2.0)
    with pytest.raises(ValueError, match="identical tracks"):
        write_stationary_sensor_capture(
            tmp_path, identity=EvaluationRunIdentity("run", "capture", "revision"),
            attributes=(item,), adapted=(replace(adapted, track_id=8),),
            checks=_checks(), configuration={}, model_invocations={}, artifact_paths={},
        )
