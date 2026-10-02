"""Durable CRA evidence records with an explicit oracle boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

from robot.evaluation.cra_experiments import RestaurantScenario
from robot.evaluation.factor_specific_cra import (
    FactorSpecificCRAReport,
    FourFactorSaturationReport,
    diagnose_factor_specific_cra,
)
from robot.evaluation.perception_readiness import (
    EvaluationScope,
    PerceptionReadinessResult,
    evaluate_perception_readiness,
)
from robot.perception.physical_attributes import EstimateStatus
from robot.risk.cra import PhysicalRiskFactor
from robot.risk.physical_factor_adapter import PhysicalFactorAdapterResult
from robot.scene.restaurant_risk_scenarios import restaurant_risk_scenario


RESTAURANT_EVIDENCE_RECORD_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


class EvaluationArtifactKind(str, Enum):
    """Kinds of immutable inputs that may affect an evaluation record."""

    CONFIG = "config"
    MODEL_WEIGHTS = "model_weights"
    DATASET_MANIFEST = "dataset_manifest"
    SCENE = "scene"
    CODE = "code"


@dataclass(frozen=True)
class EvaluationArtifactFingerprint:
    """SHA-256 identity for one configuration, model, dataset, scene, or file."""

    name: str
    kind: EvaluationArtifactKind
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "artifact name"))
        object.__setattr__(self, "kind", EvaluationArtifactKind(self.kind))
        digest = _text(self.sha256, "sha256").casefold()
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError("sha256 must contain 64 lower-case hexadecimal digits")
        object.__setattr__(self, "sha256", digest)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        name: str,
        kind: EvaluationArtifactKind,
    ) -> "EvaluationArtifactFingerprint":
        """Hash one existing file without interpreting its contents."""

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"evaluation artifact does not exist: {source}")
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return cls(name=name, kind=kind, sha256=digest.hexdigest())

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class EvaluationRunIdentity:
    """Stable identifiers for one captured evaluation input."""

    run_id: str
    capture_id: str
    code_revision: str
    artifacts: tuple[EvaluationArtifactFingerprint, ...] = ()

    def __post_init__(self) -> None:
        for name in ("run_id", "capture_id", "code_revision"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, EvaluationArtifactFingerprint) for item in artifacts):
            raise TypeError("artifacts must contain EvaluationArtifactFingerprint values")
        keys = tuple((item.kind, item.name) for item in artifacts)
        if len(set(keys)) != len(keys):
            raise ValueError("artifact fingerprints contain duplicate kind/name pairs")
        object.__setattr__(self, "artifacts", artifacts)


@dataclass(frozen=True)
class OracleScoringRecord:
    """Offline truth kept separate from runtime physical evidence."""

    scenario_id: str
    source: str
    values_json: str
    unavailable_fields: tuple[str, ...]
    available_to_runtime: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_id", _text(self.scenario_id, "scenario_id"))
        object.__setattr__(self, "source", _text(self.source, "oracle source"))
        if self.available_to_runtime:
            raise ValueError("oracle scoring values cannot be available to runtime CRA")
        try:
            values = json.loads(self.values_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("values_json must contain valid JSON") from error
        if not isinstance(values, dict):
            raise ValueError("oracle scoring values must be a JSON object")
        try:
            json.dumps(values, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("oracle scoring values must be finite JSON data") from error
        definition = restaurant_risk_scenario(self.scenario_id)
        unknown = tuple(set(values) - set(definition.oracle_scoring_fields))
        if unknown:
            raise ValueError(
                "oracle scoring fields are not declared for "
                f"{self.scenario_id}: {unknown}"
            )
        fields = tuple(self.unavailable_fields)
        if any(not isinstance(item, str) or not item for item in fields):
            raise ValueError("unavailable_fields must contain nonempty strings")
        expected_unavailable = tuple(
            field for field in definition.oracle_scoring_fields if field not in values
        )
        if fields != expected_unavailable:
            raise ValueError(
                "unavailable_fields must exactly list undeclared scoring values"
            )
        object.__setattr__(self, "unavailable_fields", fields)

    @classmethod
    def from_values(
        cls,
        scenario_id: str,
        values: Mapping[str, object] | None = None,
        *,
        source: str = "unavailable",
    ) -> "OracleScoringRecord":
        definition = restaurant_risk_scenario(scenario_id)
        provided = {} if values is None else dict(values)
        allowed = definition.oracle_scoring_fields
        if any(not isinstance(key, str) for key in provided):
            raise ValueError("oracle scoring field names must be strings")
        unknown = tuple(sorted(set(provided) - set(allowed)))
        if unknown:
            raise ValueError(
                f"oracle scoring fields are not declared for {scenario_id}: {unknown}"
            )
        try:
            payload = json.dumps(
                provided,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("oracle scoring values must be finite JSON data") from error
        return cls(
            scenario_id=definition.scenario_id,
            source=source,
            values_json=payload,
            unavailable_fields=tuple(field for field in allowed if field not in provided),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "available_to_runtime": self.available_to_runtime,
            "values": json.loads(self.values_json),
            "unavailable_fields": self.unavailable_fields,
        }


def _factor_payload(report: FactorSpecificCRAReport) -> dict[str, object]:
    return {
        "factor": report.factor.value,
        "available": report.available,
        "nominal_risk": report.nominal_risk,
        "conservative_risk": report.conservative_risk,
        "uncertainty_uplift": report.uncertainty_uplift,
        "confidence": report.confidence,
        "fallback_risk": report.fallback_risk,
        "rejection_reasons": report.rejection_reasons,
        "inputs": [
            {
                "name": item.name,
                "status": item.status,
                "source": item.source,
                "timestamp_seconds": item.timestamp_seconds,
                "confidence": item.confidence,
            }
            for item in report.inputs
        ],
    }


def _readiness_payload(
    readiness: PerceptionReadinessResult,
) -> dict[str, object]:
    return {
        "scope": readiness.scope.value,
        "ready": readiness.ready,
        "aggregate_phi_reportable": readiness.aggregate_phi_reportable,
        "factors": [
            {
                "factor": item.factor,
                "accepted": item.accepted,
                "reasons": item.reasons,
            }
            for item in readiness.factors
        ],
    }


@dataclass(frozen=True)
class RestaurantEvidenceRecord:
    """One stationary CRA evaluation input and its auditable diagnostics."""

    scenario: RestaurantScenario
    identity: EvaluationRunIdentity
    scope: EvaluationScope
    readiness: PerceptionReadinessResult
    diagnostic: FourFactorSaturationReport
    oracle_scoring: OracleScoringRecord

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, RestaurantScenario):
            raise TypeError("scenario must be a RestaurantScenario")
        if not isinstance(self.identity, EvaluationRunIdentity):
            raise TypeError("identity must be an EvaluationRunIdentity")
        object.__setattr__(self, "scope", EvaluationScope(self.scope))
        if not isinstance(self.readiness, PerceptionReadinessResult):
            raise TypeError("readiness must be a PerceptionReadinessResult")
        if not isinstance(self.diagnostic, FourFactorSaturationReport):
            raise TypeError("diagnostic must be a FourFactorSaturationReport")
        if not isinstance(self.oracle_scoring, OracleScoringRecord):
            raise TypeError("oracle_scoring must be an OracleScoringRecord")

        definition = restaurant_risk_scenario(self.scenario.scenario_id)
        if definition.factor.value != self.scenario.hazard_family:
            raise ValueError("scenario hazard family disagrees with the catalog")
        if self.readiness.scenario_id != self.scenario.scenario_id:
            raise ValueError("readiness scenario_id does not match the record")
        if self.readiness.evidence_mode is not self.scenario.evidence_mode:
            raise ValueError("readiness evidence mode does not match the record")
        if self.readiness.scope is not self.scope:
            raise ValueError("readiness scope does not match the record")
        if self.oracle_scoring.scenario_id != self.scenario.scenario_id:
            raise ValueError("oracle scoring scenario_id does not match the record")
        for factor in self.diagnostic.factors:
            for item in factor.inputs:
                try:
                    EstimateStatus(item.status)
                except ValueError as error:
                    raise ValueError(
                        f"unsupported input provenance status: {item.status!r}"
                    ) from error

    @property
    def record_id(self) -> str:
        """Return a deterministic identifier for this evaluation input."""

        key = {
            "run_id": self.identity.run_id,
            "capture_id": self.identity.capture_id,
            "scenario_id": self.scenario.scenario_id,
            "seed": self.scenario.seed,
            "evidence_mode": self.scenario.evidence_mode.value,
            "scope": self.scope.value,
            "track_id": self.diagnostic.track_id,
            "timestamp_seconds": self.diagnostic.timestamp_seconds,
        }
        encoded = json.dumps(key, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _cra_payload(self) -> dict[str, object]:
        if self.scope is EvaluationScope.FACTOR_SPECIFIC:
            factor = PhysicalRiskFactor(self.scenario.hazard_family)
            return {
                "scope": self.scope.value,
                "factor_specific": {
                    factor.value: _factor_payload(self.diagnostic.factor(factor))
                },
                "scope_note": (
                    "this record reports only the named factor; the four-factor "
                    "aggregate is intentionally omitted"
                ),
            }
        return {
            "scope": self.scope.value,
            "aggregate": {
                "reportable": self.readiness.aggregate_phi_reportable,
                "nominal_phi": self.diagnostic.nominal_phi,
                "conservative_phi": self.diagnostic.conservative_phi,
                "saturation_threshold": self.diagnostic.saturation_threshold,
                "saturated": self.diagnostic.aggregate_saturated,
                "unknown_fallback_factors": tuple(
                    item.value
                    for item in self.diagnostic.unknown_fallback_factors
                ),
                "saturation_involves_unknown_fallback": (
                    self.diagnostic.saturation_involves_unknown_fallback
                ),
            },
            "factor_specific": {
                item.factor.value: _factor_payload(item)
                for item in self.diagnostic.factors
            },
        }

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record without exposing oracle truth."""

        if not math.isfinite(self.diagnostic.timestamp_seconds):
            raise ValueError("diagnostic timestamp must be finite")
        return {
            "schema_version": RESTAURANT_EVIDENCE_RECORD_SCHEMA_VERSION,
            "record_id": self.record_id,
            "identity": {
                "run_id": self.identity.run_id,
                "capture_id": self.identity.capture_id,
                "code_revision": self.identity.code_revision,
                "artifacts": [item.to_dict() for item in self.identity.artifacts],
            },
            "scenario": {
                "scenario_id": self.scenario.scenario_id,
                "hazard_family": self.scenario.hazard_family,
                "description": self.scenario.description,
                "seed": self.scenario.seed,
                "evidence_mode": self.scenario.evidence_mode.value,
            },
            "runtime_evidence": {
                "track_id": self.diagnostic.track_id,
                "timestamp_seconds": self.diagnostic.timestamp_seconds,
                "readiness": _readiness_payload(self.readiness),
                "cra": self._cra_payload(),
            },
            "oracle_scoring": self.oracle_scoring.to_dict(),
            "boundaries": {
                "language_or_vlm_reasoning": False,
                "oracle_scoring_available_to_runtime": False,
                "missing_evidence_policy": (
                    "unavailable and fallback inputs remain explicit and are never "
                    "converted to observed zero risk"
                ),
            },
        }


def build_restaurant_evidence_record(
    scenario: RestaurantScenario,
    result: PhysicalFactorAdapterResult,
    *,
    identity: EvaluationRunIdentity,
    scope: EvaluationScope,
    oracle_scoring_values: Mapping[str, object] | None = None,
    oracle_scoring_source: str = "unavailable",
    saturation_threshold: float = 0.95,
) -> RestaurantEvidenceRecord:
    """Build one record from the same adapter output used by runtime CRA."""

    readiness = evaluate_perception_readiness(scenario, result, scope=scope)
    diagnostic = diagnose_factor_specific_cra(
        result,
        saturation_threshold=saturation_threshold,
    )
    oracle_scoring = OracleScoringRecord.from_values(
        scenario.scenario_id,
        oracle_scoring_values,
        source=oracle_scoring_source,
    )
    return RestaurantEvidenceRecord(
        scenario=scenario,
        identity=identity,
        scope=scope,
        readiness=readiness,
        diagnostic=diagnostic,
        oracle_scoring=oracle_scoring,
    )


def write_restaurant_evidence_record(
    record: RestaurantEvidenceRecord,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one record, refusing accidental replacement by default."""

    if not isinstance(record, RestaurantEvidenceRecord):
        raise TypeError("record must be a RestaurantEvidenceRecord")
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"evidence record already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output
