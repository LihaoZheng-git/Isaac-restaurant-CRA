"""Demonstrate the dependency-free CRA mathematical core.

This controlled demo uses manually supplied factor estimates.  It does not run
Isaac Sim, YOLO, Safe-SAGE, or a vision model.  The values are illustrative
hypotheses for software validation, not calibrated research results.

Run from the project root:

    python -u scripts/demos/cra_risk_math_demo.py
"""

from dataclasses import dataclass
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot.risk.cra import (
    CRAAggregation,
    CRAAssessment,
    CRAAssessmentConfig,
    CRARiskAssessor,
    PhysicalRiskFactor,
    PhysicalRiskProfile,
    PhysicalRiskVector,
    RiskFactorEstimate,
)


@dataclass(frozen=True)
class ControlledScenario:
    """One manually annotated CRA comparison case."""

    name: str
    interpretation: str
    profile: PhysicalRiskProfile


def absent(reason: str) -> RiskFactorEstimate:
    """Return one confidently absent physical factor."""

    return RiskFactorEstimate.known_absent(
        source="controlled_oracle",
        evidence=(reason,),
    )


def estimate(
    severity: float,
    interaction_likelihood: float,
    *,
    confidence: float = 1.0,
    evidence: str,
) -> RiskFactorEstimate:
    """Return one controlled physical-factor estimate."""

    return RiskFactorEstimate(
        severity=severity,
        interaction_likelihood=interaction_likelihood,
        confidence=confidence,
        source="controlled_oracle",
        evidence=(evidence,),
    )


def controlled_scenarios() -> tuple[ControlledScenario, ...]:
    """Return low-, high-, compound-, and uncertain-risk examples."""

    return (
        ControlledScenario(
            name="ordinary chair",
            interpretation="Stable furniture with small contact consequence.",
            profile=PhysicalRiskProfile(
                thermal=absent("no heat source"),
                contact=estimate(
                    0.20,
                    0.20,
                    evidence="rounded furniture surface",
                ),
                potential=estimate(
                    0.30,
                    0.20,
                    evidence="chair is supported and stable",
                ),
                kinetic=absent("chair is stationary"),
            ),
        ),
        ControlledScenario(
            name="exposed knife",
            interpretation="A severe sharp edge is accessible in the route.",
            profile=PhysicalRiskProfile(
                thermal=absent("no heat source"),
                contact=estimate(
                    0.95,
                    0.85,
                    evidence="unsheathed blade is exposed",
                ),
                potential=estimate(
                    0.15,
                    0.20,
                    evidence="knife is close to the floor",
                ),
                kinetic=absent("knife is stationary"),
            ),
        ),
        ControlledScenario(
            name="waiter carrying hot soup",
            interpretation="Thermal, spill, and motion hazards coexist.",
            profile=PhysicalRiskProfile(
                thermal=estimate(
                    0.90,
                    0.80,
                    evidence="open container is visually hot and spillable",
                ),
                contact=estimate(
                    0.20,
                    0.25,
                    evidence="container has no exposed sharp edge",
                ),
                potential=estimate(
                    0.45,
                    0.60,
                    evidence="liquid is carried above the floor",
                ),
                kinetic=estimate(
                    0.45,
                    0.70,
                    evidence="carrier is crossing the planned route",
                ),
            ),
        ),
        ControlledScenario(
            name="uncertain covered container",
            interpretation=(
                "Low-confidence visual evidence moves risk toward fallback."
            ),
            profile=PhysicalRiskProfile(
                thermal=estimate(
                    0.20,
                    0.30,
                    confidence=0.25,
                    evidence="contents and temperature are poorly observed",
                ),
                contact=absent("container has no visible sharp surface"),
                potential=estimate(
                    0.25,
                    0.30,
                    confidence=0.60,
                    evidence="support stability is partially occluded",
                ),
                kinetic=absent("container is stationary"),
            ),
        ),
    )


def print_assessment(
    scenario: ControlledScenario,
    assessment: CRAAssessment,
) -> None:
    """Print one explainable factor and aggregate report."""

    print("\n" + "-" * 78)
    print(scenario.name.upper())
    print("-" * 78)
    print(scenario.interpretation)
    print("\n  factor       severity  interact.  confidence  nominal  conservative")
    for factor in PhysicalRiskFactor:
        item = assessment.factor(factor)
        print(
            f"  {factor.value:<11} "
            f"{item.estimate.severity:>8.3f}  "
            f"{item.estimate.interaction_likelihood:>9.3f}  "
            f"{item.estimate.confidence:>10.3f}  "
            f"{item.nominal_risk:>7.3f}  "
            f"{item.conservative_risk:>12.3f}"
        )
    dominant = (
        "none"
        if assessment.dominant_factor is None
        else assessment.dominant_factor.value
    )
    print(f"\n  Nominal phi:        {assessment.nominal_phi:.4f}")
    print(f"  Conservative phi:   {assessment.phi:.4f}")
    print(f"  Uncertainty uplift: {assessment.uncertainty_uplift:.4f}")
    print(f"  Overall confidence: {assessment.overall_confidence:.4f}")
    print(f"  Dominant factor:    {dominant}")


def main() -> None:
    """Run controlled CRA mathematical comparisons."""

    print("\n" + "=" * 78)
    print("CRA MATHEMATICAL CORE - CONTROLLED ORACLE DEMO")
    print("=" * 78)
    print("  Factor risk: severity x interaction likelihood")
    print("  Aggregation: noisy-OR")
    print("  Exact distance is excluded; geometry belongs to the safety layer.")
    print("  All values below are illustrative, not calibrated measurements.")
    config = CRAAssessmentConfig(
        aggregation=CRAAggregation.NOISY_OR,
        weights=PhysicalRiskVector.ones(),
        fallback_risks=PhysicalRiskVector(
            thermal=0.75,
            contact=0.75,
            potential=0.75,
            kinetic=0.75,
        ),
    )
    assessor = CRARiskAssessor(config)
    results: dict[str, CRAAssessment] = {}
    for scenario in controlled_scenarios():
        assessment = assessor.assess(scenario.profile)
        results[scenario.name] = assessment
        print_assessment(scenario, assessment)

    ordinary = results["ordinary chair"].phi
    knife = results["exposed knife"].phi
    compound = results["waiter carrying hot soup"].phi
    uncertain = results["uncertain covered container"]
    ordering_valid = ordinary < knife < compound
    uncertainty_valid = uncertain.phi >= uncertain.nominal_phi
    bounded = all(0.0 <= item.phi <= 1.0 for item in results.values())

    print("\n" + "=" * 78)
    print("CONTROLLED MATHEMATICAL CHECKS")
    print("=" * 78)
    print(f"  Low < contact < compound risk: {ordering_valid}")
    print(f"  Uncertainty never lowers phi:  {uncertainty_valid}")
    print(f"  Every phi lies in [0, 1]:      {bounded}")
    if ordering_valid and uncertainty_valid and bounded:
        print(
            "\n  [PASS] The controlled CRA examples satisfy the initial "
            "mathematical acceptance properties."
        )
    else:
        print("\n  [WARNING] A controlled CRA mathematical check failed.")
    print("=" * 78)


if __name__ == "__main__":
    main()
