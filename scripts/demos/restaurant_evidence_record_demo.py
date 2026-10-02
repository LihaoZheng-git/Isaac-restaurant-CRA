"""Write factor-specific and four-factor records from one oracle fixture.

This checks the evaluation artifact boundary only. It does not run perception,
Isaac sensors, Safe-SAGE, or robot motion, and it makes no sensor-performance
claim.
"""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.evaluation.cra_experiments import (
    EvidenceMode,
    primary_restaurant_scenarios,
)
from robot.evaluation.perception_readiness import EvaluationScope
from robot.evaluation.restaurant_evidence_records import (
    EvaluationArtifactFingerprint,
    EvaluationArtifactKind,
    EvaluationRunIdentity,
    build_restaurant_evidence_record,
    write_restaurant_evidence_record,
)
from robot.perception.physical_attributes import (
    EstimateStatus,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.risk.physical_factor_adapter import adapt_physical_factors


OUTPUT_DIRECTORY = ROOT / "outputs" / "restaurant_evidence_records"


def _oracle_cue(value: float, source: str) -> ScalarEstimate:
    return ScalarEstimate(
        value,
        max(0.0, value - 0.1),
        min(1.0, value + 0.1),
        1.0,
        "1",
        source,
        2.0,
        EstimateStatus.ORACLE,
        ("authored record-contract fixture",),
    )


def _code_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def main() -> None:
    scenario = next(
        item
        for item in primary_restaurant_scenarios(
            evidence_mode=EvidenceMode.ORACLE,
            seed=0,
        )
        if item.scenario_id == "carried_hot_soup"
    )
    base = PhysicalAttributes.unknown(track_id=4, timestamp_seconds=2.0)
    attributes = replace(
        base,
        thermal=replace(
            base.thermal,
            heat_source_likelihood=_oracle_cue(
                0.7,
                "oracle_visible_plume_fixture",
            ),
        ),
    )
    adapted = adapt_physical_factors(
        attributes,
        timestamp_seconds=2.0,
        thermal_interaction_likelihood=_oracle_cue(
            0.4,
            "oracle_route_interaction_fixture",
        ),
    )
    identity = EvaluationRunIdentity(
        run_id="evidence-record-contract-demo",
        capture_id="oracle-frame-0001",
        code_revision=_code_revision(),
        artifacts=(
            EvaluationArtifactFingerprint.from_file(
                ROOT / "configs" / "safe_sage_paper_baseline.json",
                name="safe_sage_paper_baseline",
                kind=EvaluationArtifactKind.CONFIG,
            ),
        ),
    )
    scoring_values = {
        "plume_pose_robot_xyz_m": [2.0, 0.0, 1.1],
        "container_pose_robot_xyz_m": [2.0, 0.0, 0.65],
    }
    factor_record = build_restaurant_evidence_record(
        scenario,
        adapted,
        identity=identity,
        scope=EvaluationScope.FACTOR_SPECIFIC,
        oracle_scoring_values=scoring_values,
        oracle_scoring_source="authored_fixture",
    )
    aggregate_record = build_restaurant_evidence_record(
        scenario,
        adapted,
        identity=identity,
        scope=EvaluationScope.FOUR_FACTOR,
        oracle_scoring_values=scoring_values,
        oracle_scoring_source="authored_fixture",
    )
    factor_path = write_restaurant_evidence_record(
        factor_record,
        OUTPUT_DIRECTORY / "thermal-factor-specific.json",
        overwrite=True,
    )
    aggregate_path = write_restaurant_evidence_record(
        aggregate_record,
        OUTPUT_DIRECTORY / "four-factor-aggregate.json",
        overwrite=True,
    )
    factor_payload = factor_record.to_dict()
    aggregate_payload = aggregate_record.to_dict()
    assert "aggregate" not in factor_payload["runtime_evidence"]["cra"]
    assert factor_payload["runtime_evidence"]["readiness"]["ready"]
    assert not aggregate_payload["runtime_evidence"]["readiness"]["ready"]
    assert not aggregate_payload["runtime_evidence"]["cra"]["aggregate"][
        "reportable"
    ]
    assert not aggregate_payload["oracle_scoring"]["available_to_runtime"]
    print(
        json.dumps(
            {
                "factor_specific_record": str(factor_path),
                "factor_specific_ready": True,
                "factor_specific_aggregate_omitted": True,
                "four_factor_record": str(aggregate_path),
                "four_factor_ready": False,
                "unknown_fallback_factors": aggregate_payload[
                    "runtime_evidence"
                ]["cra"]["aggregate"]["unknown_fallback_factors"],
                "oracle_scoring_available_to_runtime": False,
                "robot_motion": "none",
                "sensor_performance_claim": "none",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
