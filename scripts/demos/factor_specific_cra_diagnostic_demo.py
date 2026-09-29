"""Show thermal evidence beside unknown-factor conservative phi saturation."""

from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.evaluation.factor_specific_cra import diagnose_factor_specific_cra
from robot.perception.physical_attributes import (
    EstimateStatus,
    PhysicalAttributes,
    ScalarEstimate,
)
from robot.risk.physical_factor_adapter import adapt_physical_factors


def oracle_cue(value: float, source: str) -> ScalarEstimate:
    return ScalarEstimate(
        value,
        max(0.0, value - 0.1),
        min(1.0, value + 0.1),
        1.0,
        "1",
        source,
        2.0,
        EstimateStatus.ORACLE,
        ("authored stationary evaluation fixture",),
    )


def main() -> None:
    base = PhysicalAttributes.unknown(track_id=4, timestamp_seconds=2.0)
    attributes = replace(
        base,
        thermal=replace(
            base.thermal,
            heat_source_likelihood=oracle_cue(0.7, "oracle_heat_fixture"),
        ),
    )
    adapted = adapt_physical_factors(
        attributes,
        timestamp_seconds=2.0,
        thermal_interaction_likelihood=oracle_cue(0.4, "oracle_route_fixture"),
    )
    report = diagnose_factor_specific_cra(adapted)
    print("STATIONARY FACTOR-SPECIFIC CRA DIAGNOSTIC: oracle fixture; no motion")
    for factor in report.factors:
        state = "available" if factor.available else "unknown"
        print(
            f"{factor.factor.value}: {state}; nominal={factor.nominal_risk:.3f}; "
            f"conservative={factor.conservative_risk:.3f}; "
            f"aggregate_phi_reportable={factor.aggregate_phi_reportable}"
        )
        for reason in factor.rejection_reasons:
            print(f"  reason: {reason}")
    print(
        f"four_factor_nominal_phi={report.nominal_phi:.3f}; "
        f"four_factor_conservative_phi={report.conservative_phi:.3f}; "
        f"saturated={report.aggregate_saturated}; "
        "scope=four_factor_aggregate"
    )
    print(
        "unknown_fallback_factors="
        + ",".join(item.value for item in report.unknown_fallback_factors)
    )
    assert report.saturation_involves_unknown_fallback
    assert report.factor("thermal").available
    print("PASS: unknown factors remained visible and were not treated as zero")


if __name__ == "__main__":
    main()
