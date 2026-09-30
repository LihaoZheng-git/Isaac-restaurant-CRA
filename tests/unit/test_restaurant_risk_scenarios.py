import pytest

from robot.risk.cra import PhysicalRiskFactor
from robot.scene.restaurant_risk_scenarios import (
    RESTAURANT_RISK_SCENARIO_SCHEMA_VERSION,
    RestaurantScenarioActor,
    ScenarioAppearanceBoundary,
    ScenarioGeometry,
    ScenarioPlacementAnchor,
    primary_restaurant_risk_scenarios,
    restaurant_risk_scenario,
    restaurant_risk_scenario_manifest,
)


def test_primary_catalog_matches_evaluation_scenarios_and_four_factors():
    scenarios = primary_restaurant_risk_scenarios()

    assert tuple(item.scenario_id for item in scenarios) == (
        "chair",
        "crossing_person",
        "elevated_object",
        "exposed_edge",
        "carried_hot_soup",
    )
    assert {item.factor for item in scenarios} == set(PhysicalRiskFactor)


def test_only_crossing_person_requires_authored_actor_motion():
    moving = tuple(
        item.scenario_id
        for item in primary_restaurant_risk_scenarios()
        if item.requires_actor_motion
    )

    assert moving == ("crossing_person",)
    actor = restaurant_risk_scenario("crossing_person").actors[0]
    assert actor.oracle_semantic_class == "person"
    assert actor.appearance_boundary is ScenarioAppearanceBoundary.PROCEDURAL_ORACLE_PROXY
    assert not restaurant_risk_scenario("crossing_person").runtime_sensor_claims


def test_potential_fixture_uses_support_to_landing_truth_not_object_height():
    scenario = restaurant_risk_scenario("elevated_object")

    assert "support_to_landing_fall_height_m" in scenario.oracle_scoring_fields
    assert "object_height_m" not in scenario.oracle_scoring_fields
    assert "object height is not fall height" in scenario.limitations


def test_contact_and_thermal_proxies_do_not_claim_unsupported_physics():
    contact = restaurant_risk_scenario("exposed_edge")
    thermal = restaurant_risk_scenario("carried_hot_soup")

    assert "physical sharpness" in contact.sensor_requirements
    assert any("not physical sharpness" in item for item in contact.limitations)
    assert "temperature_c" not in thermal.oracle_scoring_fields
    assert any("temperature" in item for item in thermal.limitations)
    assert all(not item.runtime_sensor_claims for item in (contact, thermal))


def test_manifest_labels_all_authored_information_as_oracle_only():
    payload = restaurant_risk_scenario_manifest()

    assert payload["schema_version"] == RESTAURANT_RISK_SCENARIO_SCHEMA_VERSION
    assert "offline scoring" in payload["oracle_boundary"]
    assert all(not item["runtime_sensor_claims"] for item in payload["scenarios"])
    chair = payload["scenarios"][0]["actors"][0]
    assert chair["placement_anchor"] == "bounds_base_center"


def test_existing_chair_uses_a_floor_anchor_instead_of_a_false_center():
    actor = restaurant_risk_scenario("chair").actors[0]

    assert actor.position_robot_xyz_m == (1.7, 0.0, 0.0)
    assert actor.placement_anchor is ScenarioPlacementAnchor.BOUNDS_BASE_CENTER


def test_lookup_and_actor_validation_are_strict():
    assert restaurant_risk_scenario("  carried_hot_soup ").factor is PhysicalRiskFactor.THERMAL
    with pytest.raises(KeyError):
        restaurant_risk_scenario("unknown")
    with pytest.raises(ValueError, match="absolute USD path"):
        RestaurantScenarioActor(
            "bad",
            "obstacle",
            ScenarioGeometry.EXISTING_PRIM,
            (1, 0, 0),
            "chair",
            ScenarioAppearanceBoundary.RENDERED_ASSET,
            existing_prim_path="relative/path",
        )
