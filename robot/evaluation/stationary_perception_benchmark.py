"""Persist stationary sensor captures and distinguish dataflow from accuracy.

Benchmark completion does not authorize motion. Each physical factor still
needs an independent accuracy study and the restaurant motion readiness gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from robot.evaluation.cra_experiments import EvidenceMode, primary_restaurant_scenarios
from robot.evaluation.factor_specific_cra import diagnose_factor_specific_cra
from robot.evaluation.perception_readiness import EvaluationScope
from robot.evaluation.restaurant_evidence_records import (
    EvaluationRunIdentity,
    build_restaurant_evidence_record,
    write_restaurant_evidence_record,
)
from robot.perception.physical_attributes import EstimateStatus, PhysicalAttributes
from robot.risk.physical_factor_adapter import PhysicalFactorAdapterResult


STATIONARY_BENCHMARK_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StationaryCaptureChecks:
    """Sensor and motion checks from the capture that produced the evidence."""

    sensor_pipeline_passed: bool
    robot_position_drift_m: float
    robot_yaw_drift_degrees: float
    maximum_position_drift_m: float
    maximum_yaw_drift_degrees: float
    camera_lidar_age_seconds: float
    maximum_sensor_age_seconds: float
    published_motion_count: int
    motion_sample_count: int
    matched_camera_lidar_track_count: int
    nonzero_wheel_command_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.sensor_pipeline_passed, bool):
            raise TypeError("sensor_pipeline_passed must be bool")
        for name in (
            "robot_position_drift_m", "robot_yaw_drift_degrees",
            "maximum_position_drift_m", "maximum_yaw_drift_degrees",
            "camera_lidar_age_seconds", "maximum_sensor_age_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "published_motion_count", "motion_sample_count",
            "matched_camera_lidar_track_count", "nonzero_wheel_command_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    @property
    def blockers(self) -> tuple[str, ...]:
        reasons = []
        if not self.sensor_pipeline_passed:
            reasons.append("camera/LiDAR sensor pipeline did not pass")
        if self.robot_position_drift_m > self.maximum_position_drift_m:
            reasons.append("robot position drift exceeded tolerance")
        if self.robot_yaw_drift_degrees > self.maximum_yaw_drift_degrees:
            reasons.append("robot yaw drift exceeded tolerance")
        if self.camera_lidar_age_seconds > self.maximum_sensor_age_seconds:
            reasons.append("camera/LiDAR sensor age exceeded tolerance")
        if self.nonzero_wheel_command_count:
            reasons.append("nonzero wheel command invalidates stationary capture")
        if self.published_motion_count:
            reasons.append("stationary scene published nonzero object motion")
        if not self.motion_sample_count:
            reasons.append("stationary motion samples unavailable")
        if not self.matched_camera_lidar_track_count:
            reasons.append("no accepted camera/LiDAR association")
        return tuple(reasons)


def _scalar_is_sensor(estimate) -> bool:
    return (
        estimate.is_available
        and estimate.confidence > 0
        and estimate.status in {EstimateStatus.MEASURED, EstimateStatus.INFERRED}
    )


def write_stationary_sensor_capture(
    output_directory: str | Path,
    *,
    identity: EvaluationRunIdentity,
    attributes: tuple[PhysicalAttributes, ...],
    adapted: tuple[PhysicalFactorAdapterResult, ...],
    checks: StationaryCaptureChecks,
    configuration: dict[str, object],
    model_invocations: dict[str, object],
    artifact_paths: dict[str, str],
    oracle_scoring: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write raw evidence plus scenario-scoped records after sensor inference.

    Authored bounds are optional offline scoring only. They never participate
    in adapter construction, dataflow acceptance, or factor readiness.
    """

    if not isinstance(identity, EvaluationRunIdentity):
        raise TypeError("identity must be EvaluationRunIdentity")
    if not isinstance(checks, StationaryCaptureChecks):
        raise TypeError("checks must be StationaryCaptureChecks")
    attribute_by_id = {item.track_id: item for item in attributes}
    result_by_id = {item.track_id: item for item in adapted}
    if len(attribute_by_id) != len(attributes) or len(result_by_id) != len(adapted):
        raise ValueError("capture contains duplicate track IDs")
    if attribute_by_id.keys() != result_by_id.keys():
        raise ValueError("physical attributes and adapter results must have identical tracks")
    output = Path(output_directory)
    if (output / "capture.json").exists():
        raise FileExistsError("stationary capture already exists; use a fresh capture directory")
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    record_paths = []
    chair = next(
        item for item in primary_restaurant_scenarios(evidence_mode=EvidenceMode.SENSOR)
        if item.scenario_id == "chair"
    )
    for item in attributes:
        result = result_by_id[item.track_id]
        if result.timestamp_seconds != item.timestamp_seconds:
            raise ValueError("attribute and adapter timestamps must match")
        dimensions = (
            item.geometry.width_m, item.geometry.depth_m, item.geometry.height_m,
        )
        diagnostic = diagnose_factor_specific_cra(result)
        summaries.append({
            "track_id": item.track_id,
            "semantic_class": item.semantic_class,
            "sensor_geometry_available": all(_scalar_is_sensor(v) for v in dimensions),
            "sensor_mass_available": _scalar_is_sensor(item.mass.mass_kg),
            "physical_attributes": asdict(item),
            "adapter_configuration": asdict(result.config),
            "factor_diagnostics": diagnostic.to_dict(),
        })
        # A chair observation cannot be relabeled as a hot-soup or edge scenario
        # merely to produce a factor record. Other classes retain full diagnostics.
        if item.semantic_class == "chair" and item.semantic_source != "unavailable":
            for scope in EvaluationScope:
                record = build_restaurant_evidence_record(
                    chair, result, identity=identity, scope=scope,
                )
                path = output / f"track-{item.track_id}-{scope.value}.json"
                write_restaurant_evidence_record(record, path)
                record_paths.append(str(path.resolve()))
    payload = {
        "schema_version": STATIONARY_BENCHMARK_SCHEMA_VERSION,
        "run_id": identity.run_id,
        "capture_id": identity.capture_id,
        "code_revision": identity.code_revision,
        "artifacts": [item.to_dict() for item in identity.artifacts],
        "evidence_mode": EvidenceMode.SENSOR.value,
        "checks": asdict(checks),
        "capture_blockers": checks.blockers,
        "capture_passed": not checks.blockers,
        "configuration": configuration,
        "model_invocations": model_invocations,
        "artifact_paths": artifact_paths,
        "tracks": summaries,
        "scenario_record_paths": record_paths,
        "oracle_scoring": {
            "available_to_runtime": False,
            "values": oracle_scoring or {},
        },
        "accuracy_boundary": (
            "sensor dataflow and stationary stability only; no independent mass, "
            "contact consequence, or thermal severity reference is supplied"
        ),
    }
    (output / "capture.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def summarize_stationary_captures(
    captures: tuple[dict[str, object], ...],
    *,
    minimum_captures: int = 3,
) -> dict[str, object]:
    """Report repeated sensor dataflow without granting factor accuracy."""

    if isinstance(minimum_captures, bool) or not isinstance(minimum_captures, int) or minimum_captures < 1:
        raise ValueError("minimum_captures must be a positive integer")
    identities = tuple((item["run_id"], item["capture_id"]) for item in captures)
    if len(set(identities)) != len(identities):
        raise ValueError("capture identities must be unique")
    blockers = []
    if len(captures) < minimum_captures:
        blockers.append("insufficient independent capture runs")
    counts = {"geometry": 0, "visual_mass": 0, "contact_invocation": 0}
    maximum_drift = 0.0
    for item in captures:
        if item.get("schema_version") != STATIONARY_BENCHMARK_SCHEMA_VERSION:
            raise ValueError("unsupported stationary benchmark schema")
        if item.get("evidence_mode") != EvidenceMode.SENSOR.value:
            raise ValueError("stationary sensor summary cannot pool oracle/prior captures")
        checks = StationaryCaptureChecks(**item["checks"])
        if not item["capture_passed"] or checks.blockers:
            blockers.append(f"capture {item['capture_id']} failed stationary checks")
        if not item["artifacts"]:
            blockers.append(f"capture {item['capture_id']} lacks input fingerprints")
        tracks = item["tracks"]
        counts["geometry"] += int(any(row["sensor_geometry_available"] for row in tracks))
        counts["visual_mass"] += int(any(row["sensor_mass_available"] for row in tracks))
        contact = item["model_invocations"].get("contact", {})
        counts["contact_invocation"] += int(contact.get("invoked", False) and contact.get("pipeline_valid", False))
        maximum_drift = max(maximum_drift, item["checks"]["robot_position_drift_m"])
    for component, count in counts.items():
        if count < minimum_captures:
            blockers.append(f"{component}: insufficient validated capture runs")
    signatures = {
        tuple(sorted((v["kind"], v["name"], v["sha256"]) for v in item["artifacts"]))
        for item in captures
    }
    if len(signatures) > 1:
        blockers.append("capture artifact fingerprints differ; compare separately")
    accuracy_blockers = (
        "visual mass: live restaurant reference masses and domain calibration unavailable",
        "potential: positive live fall-height validation, harmful-fall interaction, and mass calibration incomplete",
        "kinetic: slow, occluded, and multiple-object motion coverage incomplete; nominal controlled test is separate",
        "contact: general geometry, physical sharpness, material consequence, exposure, and route orientation incomplete",
        "thermal: no verified local plume checkpoint or sensor-domain score calibration",
    )
    return {
        "schema_version": STATIONARY_BENCHMARK_SCHEMA_VERSION,
        "capture_count": len(captures),
        "minimum_captures": minimum_captures,
        "validated_capture_counts": counts,
        "maximum_robot_position_drift_m": maximum_drift,
        "dataflow_passed": not blockers,
        "dataflow_blockers": blockers,
        "factor_accuracy_blockers": accuracy_blockers,
        "ready_for_sensor_motion": False,
        "scope": "repeated stationary restaurant sensor dataflow",
    }
