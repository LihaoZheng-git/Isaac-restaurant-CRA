from dataclasses import FrozenInstanceError

import pytest

from robot.risk.cra import CRARiskAssessor, PhysicalRiskFactor
from robot.risk.cra_oracle_scenarios import (
    ORACLE_SOURCE,
    controlled_cra_oracle_scenarios,
    cra_oracle_scenario,
)


def test_catalog_has_stable_unique_order() -> None:
    scenarios = controlled_cra_oracle_scenarios()

    assert tuple(scenario.key for scenario in scenarios) == (
        "ordinary_chair",
        "fast_crossing_person",
        "exposed_knife",
        "hot_soup_waiter",
    )
    assert len({scenario.key for scenario in scenarios}) == len(scenarios)


def test_each_scenario_has_the_expected_dominant_factor() -> None:
    assessor = CRARiskAssessor()

    for scenario in controlled_cra_oracle_scenarios():
        assessment = assessor.assess(scenario.profile)
        assert assessment.dominant_factor is scenario.expected_dominant_factor


def test_catalog_spans_all_four_physical_factors() -> None:
    dominant_factors = {
        scenario.expected_dominant_factor
        for scenario in controlled_cra_oracle_scenarios()
    }

    assert dominant_factors == set(PhysicalRiskFactor)


def test_controlled_profiles_remove_uncertainty_as_a_confound() -> None:
    for scenario in controlled_cra_oracle_scenarios():
        for _, estimate in scenario.profile.items():
            assert estimate.confidence == pytest.approx(1.0)
            assert estimate.source == ORACLE_SOURCE


def test_catalog_risks_are_distinct_and_ordered_for_the_demo() -> None:
    assessor = CRARiskAssessor()
    risks = tuple(
        assessor.assess(scenario.profile).phi
        for scenario in controlled_cra_oracle_scenarios()
    )

    assert risks == tuple(sorted(risks))
    assert len(set(risks)) == len(risks)


def test_lookup_normalizes_case_and_whitespace() -> None:
    scenario = cra_oracle_scenario("  EXPOSED_KNIFE ")

    assert scenario.key == "exposed_knife"


def test_unknown_scenario_key_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown CRA oracle scenario"):
        cra_oracle_scenario("broken_glass")


def test_scenario_is_immutable() -> None:
    scenario = controlled_cra_oracle_scenarios()[0]

    with pytest.raises(FrozenInstanceError):
        scenario.display_name = "changed"  # type: ignore[misc]
