import json

import pytest

from robot.safety.class_safety_policy import (
    ClassSafetyParameters,
    ClassSafetyPolicy,
    LinearRiskFluxMapping,
)
from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.semantic_obstacle import SemanticObstacle


def default_config() -> dict[str, object]:
    """Return a small deterministic class-policy configuration."""

    return {
        "schema_version": 2,
        "unknown_class": "unknown",
        "risk_flux_mapping": {
            "type": "linear",
            "minimum_magnitude": 1.0,
            "maximum_magnitude": 6.0,
        },
        "classes": {
            "unknown": {
                "risk": 1.0,
            },
            "wall": {
                "risk": 0.0,
            },
            "chair": {
                "risk": 0.4,
            },
            "person": {
                "risk": 1.0,
            },
        },
    }


def write_config(
    tmp_path,
    config: object,
):
    """Write a JSON policy and return its path."""

    config_path = tmp_path / "class_safety.json"
    config_path.write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    return config_path


def make_semantic_obstacle(
    semantic_class: str,
) -> SemanticObstacle:
    """Return one deterministic semantic obstacle."""

    track = TrackedObstacle(
        track_id=4,
        centroid_world_xy=(28.2, 5.8),
        centroid_robot_xy=(1.7, 0.8),
        velocity_world_xy=(0.0, -0.6),
        velocity_robot_xy=(0.0, -0.6),
        point_count=9,
        observation_count=12,
        matched_previous_track=True,
    )
    return SemanticObstacle(
        tracked_obstacle=track,
        semantic_class=semantic_class,
        semantic_confidence=1.0,
    )


def test_parameters_return_negative_signed_boundary_flux() -> None:
    parameters = ClassSafetyParameters(
        risk=0.4,
        boundary_flux_magnitude=3,
    )

    assert parameters.risk == pytest.approx(0.4)
    assert parameters.boundary_flux_magnitude == pytest.approx(3.0)
    assert parameters.boundary_flux == pytest.approx(-3.0)


def test_linear_mapping_reproduces_expected_fluxes() -> None:
    mapping = LinearRiskFluxMapping(
        minimum_magnitude=1.0,
        maximum_magnitude=6.0,
    )

    assert mapping.magnitude_for_risk(0.0) == pytest.approx(1.0)
    assert mapping.magnitude_for_risk(0.4) == pytest.approx(3.0)
    assert mapping.magnitude_for_risk(1.0) == pytest.approx(6.0)


def test_policy_loads_expected_class_ordering(tmp_path) -> None:
    policy = ClassSafetyPolicy.from_json(
        write_config(tmp_path, default_config())
    )

    wall = policy.parameters_for("wall")
    chair = policy.parameters_for("chair")
    person = policy.parameters_for("person")

    assert wall.risk < chair.risk < person.risk
    assert (
        wall.boundary_flux_magnitude
        < chair.boundary_flux_magnitude
        < person.boundary_flux_magnitude
    )
    assert policy.known_classes == (
        "chair",
        "person",
        "unknown",
        "wall",
    )
    assert policy.risk_flux_mapping is not None
    assert policy.risk_flux_mapping.minimum_magnitude == pytest.approx(1.0)
    assert policy.risk_flux_mapping.maximum_magnitude == pytest.approx(6.0)


def test_lookup_normalizes_class_name(tmp_path) -> None:
    policy = ClassSafetyPolicy.from_json(
        write_config(tmp_path, default_config())
    )

    parameters = policy.parameters_for("  PERSON ")

    assert parameters.risk == pytest.approx(1.0)
    assert parameters.boundary_flux == pytest.approx(-6.0)


def test_unconfigured_class_uses_conservative_fallback(tmp_path) -> None:
    policy = ClassSafetyPolicy.from_json(
        write_config(tmp_path, default_config())
    )

    fallback = policy.parameters_for("fork_on_floor")

    assert fallback is policy.class_parameters["unknown"]
    assert fallback.risk == pytest.approx(1.0)
    assert fallback.boundary_flux_magnitude == pytest.approx(6.0)


def test_policy_accepts_semantic_obstacle(tmp_path) -> None:
    policy = ClassSafetyPolicy.from_json(
        write_config(tmp_path, default_config())
    )

    parameters = policy.parameters_for_obstacle(
        make_semantic_obstacle("chair")
    )

    assert parameters.risk == pytest.approx(0.4)
    assert parameters.boundary_flux == pytest.approx(-3.0)


def test_policy_mapping_is_read_only(tmp_path) -> None:
    policy = ClassSafetyPolicy.from_json(
        write_config(tmp_path, default_config())
    )

    with pytest.raises(TypeError):
        policy.class_parameters["chair"] = ClassSafetyParameters(
            risk=0.5,
            boundary_flux_magnitude=4.0,
        )


@pytest.mark.parametrize(
    "risk",
    [-0.01, 1.01, float("inf"), float("nan")],
)
def test_invalid_risk_is_rejected(risk: float) -> None:
    with pytest.raises(ValueError):
        ClassSafetyParameters(
            risk=risk,
            boundary_flux_magnitude=1.0,
        )


@pytest.mark.parametrize("risk", [True, "high", None])
def test_non_numeric_risk_is_rejected(risk: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        ClassSafetyParameters(
            risk=risk,
            boundary_flux_magnitude=1.0,
        )


@pytest.mark.parametrize(
    "flux_magnitude",
    [0.0, -1.0, float("inf"), float("nan")],
)
def test_invalid_flux_magnitude_is_rejected(
    flux_magnitude: float,
) -> None:
    with pytest.raises(ValueError):
        ClassSafetyParameters(
            risk=0.5,
            boundary_flux_magnitude=flux_magnitude,
        )


def test_unknown_class_entry_is_required() -> None:
    with pytest.raises(ValueError, match="unknown_class"):
        ClassSafetyPolicy(
            {
                "chair": ClassSafetyParameters(
                    risk=0.4,
                    boundary_flux_magnitude=3.0,
                )
            }
        )


def test_duplicate_normalized_class_is_rejected() -> None:
    parameters = ClassSafetyParameters(
        risk=0.4,
        boundary_flux_magnitude=3.0,
    )

    with pytest.raises(ValueError, match="duplicate"):
        ClassSafetyPolicy(
            {
                "unknown": parameters,
                "Dining Table": parameters,
                "dining-table": parameters,
            }
        )


def test_wrong_schema_version_is_rejected(tmp_path) -> None:
    config = default_config()
    config["schema_version"] = 1

    with pytest.raises(ValueError, match="schema_version"):
        ClassSafetyPolicy.from_json(write_config(tmp_path, config))


def test_missing_class_field_is_rejected(tmp_path) -> None:
    config = default_config()
    classes = config["classes"]
    classes["chair"] = {}

    with pytest.raises(ValueError, match="missing"):
        ClassSafetyPolicy.from_json(write_config(tmp_path, config))


def test_legacy_independent_flux_field_is_rejected(tmp_path) -> None:
    config = default_config()
    classes = config["classes"]
    classes["chair"] = {
        "risk": 0.4,
        "boundary_flux_magnitude": 3.0,
    }

    with pytest.raises(ValueError, match="unexpected"):
        ClassSafetyPolicy.from_json(write_config(tmp_path, config))


@pytest.mark.parametrize(
    "mapping_values",
    (
        {"minimum_magnitude": 0.0, "maximum_magnitude": 6.0},
        {"minimum_magnitude": 7.0, "maximum_magnitude": 6.0},
    ),
)
def test_invalid_risk_flux_mapping_is_rejected(
    mapping_values: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        LinearRiskFluxMapping(**mapping_values)


def test_non_semantic_obstacle_is_rejected(tmp_path) -> None:
    policy = ClassSafetyPolicy.from_json(
        write_config(tmp_path, default_config())
    )

    with pytest.raises(TypeError, match="SemanticObstacle"):
        policy.parameters_for_obstacle("chair")
