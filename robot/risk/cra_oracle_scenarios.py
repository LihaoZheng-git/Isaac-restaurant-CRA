"""Controlled CRA profiles for software and simulator validation.

The profiles in this module are deliberately hand-authored counterfactuals.
They let tests hold perception and geometry fixed while changing only the
four physical-risk hypotheses consumed by CRA.  They are not calibrated
measurements and must not be presented as learned perception output.
"""

from dataclasses import dataclass

from robot.risk.cra import (
    PhysicalRiskFactor,
    PhysicalRiskProfile,
    RiskFactorEstimate,
)


ORACLE_SOURCE = "controlled_oracle"


def _absent(reason: str) -> RiskFactorEstimate:
    """Return one confidently absent factor for a controlled scenario."""

    return RiskFactorEstimate.known_absent(
        source=ORACLE_SOURCE,
        evidence=(reason,),
    )


def _estimate(
    severity: float,
    interaction_likelihood: float,
    *,
    evidence: str,
) -> RiskFactorEstimate:
    """Return one high-confidence oracle factor estimate."""

    return RiskFactorEstimate(
        severity=severity,
        interaction_likelihood=interaction_likelihood,
        confidence=1.0,
        source=ORACLE_SOURCE,
        evidence=(evidence,),
    )


@dataclass(frozen=True)
class CRAOracleScenario:
    """One named, manually specified CRA validation hypothesis."""

    key: str
    display_name: str
    description: str
    profile: PhysicalRiskProfile
    expected_dominant_factor: PhysicalRiskFactor

    def __post_init__(self) -> None:
        for name in ("key", "display_name", "description"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string.")
            normalized = value.strip()
            if not normalized:
                raise ValueError(f"{name} cannot be empty.")
            object.__setattr__(self, name, normalized)
        if not isinstance(self.profile, PhysicalRiskProfile):
            raise TypeError("profile must be a PhysicalRiskProfile.")
        try:
            dominant = PhysicalRiskFactor(self.expected_dominant_factor)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "expected_dominant_factor must be a PhysicalRiskFactor."
            ) from error
        object.__setattr__(self, "expected_dominant_factor", dominant)


def ordinary_chair_profile() -> PhysicalRiskProfile:
    """Return a low-risk, stable furniture hypothesis."""

    return PhysicalRiskProfile(
        thermal=_absent("no heat source is present"),
        contact=_estimate(
            0.20,
            0.20,
            evidence="rounded furniture surfaces are accessible",
        ),
        potential=_estimate(
            0.30,
            0.20,
            evidence="the chair is supported and stable",
        ),
        kinetic=_absent("the chair is stationary"),
    )


def fast_crossing_person_profile() -> PhysicalRiskProfile:
    """Return a kinetic-risk hypothesis for a crossing pedestrian."""

    return PhysicalRiskProfile(
        thermal=_absent("no thermal hazard is observed"),
        contact=_estimate(
            0.35,
            0.40,
            evidence="a person may enter the robot's swept path",
        ),
        potential=_estimate(
            0.15,
            0.20,
            evidence="loss of balance is possible but not prominent",
        ),
        kinetic=_estimate(
            0.75,
            0.80,
            evidence="the person is crossing quickly near the route",
        ),
    )


def exposed_knife_profile() -> PhysicalRiskProfile:
    """Return a high contact-risk hypothesis for an exposed blade."""

    return PhysicalRiskProfile(
        thermal=_absent("no heat source is present"),
        contact=_estimate(
            0.95,
            0.85,
            evidence="an unsheathed blade is exposed",
        ),
        potential=_estimate(
            0.15,
            0.20,
            evidence="the knife is close to the floor",
        ),
        kinetic=_absent("the knife is stationary"),
    )


def hot_soup_waiter_profile() -> PhysicalRiskProfile:
    """Return a mixed thermal/potential/kinetic restaurant hypothesis."""

    return PhysicalRiskProfile(
        thermal=_estimate(
            0.90,
            0.80,
            evidence="uncovered hot liquid could cause a burn",
        ),
        contact=_estimate(
            0.20,
            0.25,
            evidence="the carried vessel has no exposed sharp edge",
        ),
        potential=_estimate(
            0.45,
            0.60,
            evidence="the carried liquid can spill from above the floor",
        ),
        kinetic=_estimate(
            0.45,
            0.70,
            evidence="the waiter is moving across the robot route",
        ),
    )


def controlled_cra_oracle_scenarios() -> tuple[CRAOracleScenario, ...]:
    """Return the stable four-scenario pre-vision validation catalog."""

    return (
        CRAOracleScenario(
            key="ordinary_chair",
            display_name="Ordinary chair",
            description="Stable furniture with modest contact and potential risk.",
            profile=ordinary_chair_profile(),
            expected_dominant_factor=PhysicalRiskFactor.POTENTIAL,
        ),
        CRAOracleScenario(
            key="fast_crossing_person",
            display_name="Fast crossing person",
            description="A pedestrian whose dominant concern is kinetic risk.",
            profile=fast_crossing_person_profile(),
            expected_dominant_factor=PhysicalRiskFactor.KINETIC,
        ),
        CRAOracleScenario(
            key="exposed_knife",
            display_name="Exposed knife",
            description="A stationary object dominated by sharp-contact risk.",
            profile=exposed_knife_profile(),
            expected_dominant_factor=PhysicalRiskFactor.CONTACT,
        ),
        CRAOracleScenario(
            key="hot_soup_waiter",
            display_name="Waiter carrying hot soup",
            description="A mixed restaurant hazard dominated by thermal risk.",
            profile=hot_soup_waiter_profile(),
            expected_dominant_factor=PhysicalRiskFactor.THERMAL,
        ),
    )


def cra_oracle_scenario(key: str) -> CRAOracleScenario:
    """Return one controlled scenario by stable key."""

    if not isinstance(key, str):
        raise TypeError("key must be a string.")
    normalized = key.strip().casefold()
    for scenario in controlled_cra_oracle_scenarios():
        if scenario.key.casefold() == normalized:
            return scenario
    available = ", ".join(
        scenario.key for scenario in controlled_cra_oracle_scenarios()
    )
    raise KeyError(f"Unknown CRA oracle scenario {key!r}; choose from {available}.")


__all__ = [
    "CRAOracleScenario",
    "ORACLE_SOURCE",
    "controlled_cra_oracle_scenarios",
    "cra_oracle_scenario",
    "exposed_knife_profile",
    "fast_crossing_person_profile",
    "hot_soup_waiter_profile",
    "ordinary_chair_profile",
]
