import json

import pytest

from robot.evaluation.cra_experiments import (
    EvidenceMode,
    primary_restaurant_scenarios,
)
from robot.evaluation.perception_readiness import EvaluationScope
from robot.evaluation.restaurant_evidence_records import (
    EvaluationArtifactFingerprint,
    EvaluationArtifactKind,
    EvaluationRunIdentity,
    OracleScoringRecord,
    build_restaurant_evidence_record,
    write_restaurant_evidence_record,
)
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate
from robot.risk.cra import PhysicalRiskProfile, RiskFactorEstimate
from robot.risk.physical_factor_adapter import (
    PhysicalFactorAdapterConfig,
    PhysicalFactorAdapterResult,
    PhysicalFactorDiagnostic,
)


def _scenario(scenario_id="carried_hot_soup", mode=EvidenceMode.SENSOR):
    return next(
        item
        for item in primary_restaurant_scenarios(evidence_mode=mode, seed=7)
        if item.scenario_id == scenario_id
    )


def _scalar(status: EstimateStatus) -> ScalarEstimate:
    if status is EstimateStatus.UNAVAILABLE:
        return ScalarEstimate.unavailable(
            unit="1",
            source="camera_cue_unavailable",
            timestamp_seconds=3.0,
            evidence=("no supported cue",),
        )
    return ScalarEstimate(
        0.6,
        0.5,
        0.7,
        0.8,
        "1",
        "stationary_camera_fixture",
        3.0,
        status,
        ("bounded stationary observation",),
    )


def _diagnostic(name: str, status: EstimateStatus) -> PhysicalFactorDiagnostic:
    cue = _scalar(status)
    available = status is not EstimateStatus.UNAVAILABLE
    estimate = (
        RiskFactorEstimate(
            0.6,
            0.5,
            0.8,
            source="stationary_factor_adapter",
            evidence=("bounded stationary observation",),
        )
        if available
        else RiskFactorEstimate.unknown(
            source="factor_evidence_unavailable",
            evidence=("no supported cue",),
        )
    )
    energy = ScalarEstimate.unavailable(
        unit="J",
        source="energy_unavailable",
        timestamp_seconds=3.0,
    )
    rejection_reasons = () if available else ("cue: unavailable",)
    return PhysicalFactorDiagnostic(
        name,
        (("cue", cue),),
        energy,
        estimate,
        rejection_reasons,
    )


def _adapter_result(
    *,
    thermal=EstimateStatus.INFERRED,
    contact=EstimateStatus.INFERRED,
    potential=EstimateStatus.INFERRED,
    kinetic=EstimateStatus.INFERRED,
) -> PhysicalFactorAdapterResult:
    diagnostics = {
        "thermal": _diagnostic("thermal", thermal),
        "contact": _diagnostic("contact", contact),
        "potential": _diagnostic("potential", potential),
        "kinetic": _diagnostic("kinetic", kinetic),
    }
    return PhysicalFactorAdapterResult(
        track_id=12,
        timestamp_seconds=3.0,
        config=PhysicalFactorAdapterConfig(),
        profile=PhysicalRiskProfile(
            diagnostics["thermal"].estimate,
            diagnostics["contact"].estimate,
            diagnostics["potential"].estimate,
            diagnostics["kinetic"].estimate,
        ),
        potential=diagnostics["potential"],
        kinetic=diagnostics["kinetic"],
        contact=diagnostics["contact"],
        thermal=diagnostics["thermal"],
    )


def _identity() -> EvaluationRunIdentity:
    return EvaluationRunIdentity(
        run_id="stationary-review-001",
        capture_id="frame-0007",
        code_revision="0123456789abcdef",
    )


def test_factor_specific_record_omits_aggregate_and_keeps_oracle_separate():
    result = _adapter_result(
        contact=EstimateStatus.UNAVAILABLE,
        potential=EstimateStatus.UNAVAILABLE,
        kinetic=EstimateStatus.UNAVAILABLE,
    )
    record = build_restaurant_evidence_record(
        _scenario(),
        result,
        identity=_identity(),
        scope=EvaluationScope.FACTOR_SPECIFIC,
        oracle_scoring_values={
            "plume_pose_robot_xyz_m": [2.0, 0.0, 1.1],
            "container_pose_robot_xyz_m": [2.0, 0.0, 0.65],
        },
        oracle_scoring_source="authored_fixture",
    )

    payload = record.to_dict()
    runtime = payload["runtime_evidence"]
    assert runtime["readiness"]["ready"]
    assert runtime["readiness"]["aggregate_phi_reportable"] is False
    assert "aggregate" not in runtime["cra"]
    assert tuple(runtime["cra"]["factor_specific"]) == ("thermal",)
    assert runtime["cra"]["factor_specific"]["thermal"]["inputs"][0][
        "status"
    ] == "inferred"
    assert payload["oracle_scoring"]["available_to_runtime"] is False
    assert payload["oracle_scoring"]["values"][
        "plume_pose_robot_xyz_m"
    ] == [2.0, 0.0, 1.1]


def test_four_factor_record_reports_phi_only_when_all_sensor_inputs_are_ready():
    record = build_restaurant_evidence_record(
        _scenario(),
        _adapter_result(),
        identity=_identity(),
        scope=EvaluationScope.FOUR_FACTOR,
    )

    payload = record.to_dict()
    assert payload["runtime_evidence"]["readiness"]["ready"]
    aggregate = payload["runtime_evidence"]["cra"]["aggregate"]
    assert aggregate["reportable"] is True
    assert 0 <= aggregate["nominal_phi"] <= aggregate["conservative_phi"] <= 1
    assert set(payload["runtime_evidence"]["cra"]["factor_specific"]) == {
        "thermal",
        "contact",
        "potential",
        "kinetic",
    }


def test_unknown_inputs_remain_blocking_and_visible_in_aggregate_record():
    record = build_restaurant_evidence_record(
        _scenario(),
        _adapter_result(
            contact=EstimateStatus.UNAVAILABLE,
            potential=EstimateStatus.UNAVAILABLE,
            kinetic=EstimateStatus.UNAVAILABLE,
        ),
        identity=_identity(),
        scope=EvaluationScope.FOUR_FACTOR,
    )

    payload = record.to_dict()
    aggregate = payload["runtime_evidence"]["cra"]["aggregate"]
    assert payload["runtime_evidence"]["readiness"]["ready"] is False
    assert aggregate["reportable"] is False
    assert set(aggregate["unknown_fallback_factors"]) == {
        "contact",
        "potential",
        "kinetic",
    }
    assert aggregate["saturation_involves_unknown_fallback"]
    assert payload["runtime_evidence"]["cra"]["factor_specific"]["contact"][
        "available"
    ] is False


def test_sensor_record_rejects_oracle_input_for_runtime_readiness():
    record = build_restaurant_evidence_record(
        _scenario(),
        _adapter_result(
            thermal=EstimateStatus.ORACLE,
            contact=EstimateStatus.ORACLE,
            potential=EstimateStatus.ORACLE,
            kinetic=EstimateStatus.ORACLE,
        ),
        identity=_identity(),
        scope=EvaluationScope.FOUR_FACTOR,
    )

    payload = record.to_dict()
    assert payload["runtime_evidence"]["readiness"]["ready"] is False
    assert payload["runtime_evidence"]["cra"]["aggregate"]["reportable"] is False


def test_oracle_scoring_schema_rejects_unknown_or_nonfinite_fields():
    with pytest.raises(ValueError, match="not declared"):
        OracleScoringRecord.from_values(
            "elevated_object",
            {"object_height_m": 0.3},
            source="fixture",
        )
    with pytest.raises(ValueError, match="finite JSON"):
        OracleScoringRecord.from_values(
            "elevated_object",
            {"support_height_m": float("nan")},
            source="fixture",
        )
    with pytest.raises(ValueError, match="field names"):
        OracleScoringRecord.from_values(
            "elevated_object",
            {1: 0.8},
            source="fixture",
        )
    with pytest.raises(ValueError, match="exactly list"):
        OracleScoringRecord(
            "elevated_object",
            "fixture",
            '{"support_height_m":0.8}',
            (),
        )


def test_artifact_fingerprint_and_atomic_record_write(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"lambda":1.25}\n', encoding="utf-8")
    artifact = EvaluationArtifactFingerprint.from_file(
        config,
        name="primary_safe_sage",
        kind=EvaluationArtifactKind.CONFIG,
    )
    identity = EvaluationRunIdentity(
        "run",
        "capture",
        "revision",
        (artifact,),
    )
    record = build_restaurant_evidence_record(
        _scenario(),
        _adapter_result(),
        identity=identity,
        scope=EvaluationScope.FOUR_FACTOR,
    )
    output = tmp_path / "record.json"

    write_restaurant_evidence_record(record, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["record_id"] == record.record_id
    assert payload["identity"]["artifacts"][0]["sha256"] == artifact.sha256
    with pytest.raises(FileExistsError):
        write_restaurant_evidence_record(record, output)
