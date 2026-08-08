import json
from pathlib import Path

import pytest

from robot.safety.class_safety_policy import ClassSafetyParameters
from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.safe_sage_guidance_policy import (
    PassingSide,
    SafeSAGEGuidanceParameters,
    SafeSAGEGuidancePolicy,
)
from robot.safety.semantic_obstacle import SemanticObstacle

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_config() -> dict[str, object]:
    """Return a deterministic tangential-guidance configuration."""

    return {
        "schema_version": 1,
        "unknown_class": "unknown",
        "risk_flux_mapping": {
            "type": "linear",
            "minimum_magnitude": 1.0,
            "maximum_magnitude": 6.0,
        },
        "classes": {
            "unknown": {
                "risk": 1.0,
                "tangential_strength": 1.25,
                "passing_side": "right",
            },
            "wall": {
                "risk": 0.0,
                "tangential_strength": 0.0,
                "passing_side": "right",
            },
            "chair": {
                "risk": 0.4,
                "tangential_strength": 0.75,
                "passing_side": "right",
            },
            "person": {
                "risk": 1.0,
                "tangential_strength": 1.25,
                "passing_side": "right",
            },
        },
    }


def write_config(tmp_path, config: object):
    """Write one JSON guidance policy and return its path."""

    config_path = tmp_path / "safe_sage_class_safety.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
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


def make_parameters(
    *,
    strength: float = 0.75,
    side: PassingSide | str = PassingSide.RIGHT,
) -> SafeSAGEGuidanceParameters:
    """Return one deterministic materialized parameter object."""

    return SafeSAGEGuidanceParameters(
        safety_parameters=ClassSafetyParameters(
            risk=0.4,
            boundary_flux_magnitude=3.0,
        ),
        tangential_strength=strength,
        passing_side=side,
    )


def test_materialized_parameters_expose_normal_and_tangential_data() -> None:
    parameters = make_parameters()

    assert parameters.risk == pytest.approx(0.4)
    assert parameters.boundary_flux_magnitude == pytest.approx(3.0)
    assert parameters.boundary_flux == pytest.approx(-3.0)
    assert parameters.tangential_strength == pytest.approx(0.75)
    assert parameters.passing_side is PassingSide.RIGHT
    assert parameters.has_tangential_guidance


def test_zero_strength_disables_tangential_guidance() -> None:
    parameters = make_parameters(strength=0.0)

    assert not parameters.has_tangential_guidance


def test_passing_side_is_normalized() -> None:
    parameters = make_parameters(side=" LEFT ")

    assert parameters.passing_side is PassingSide.LEFT


def test_policy_loads_expected_class_parameters(tmp_path) -> None:
    policy = SafeSAGEGuidancePolicy.from_json(
        write_config(tmp_path, default_config())
    )

    wall = policy.parameters_for("wall")
    chair = policy.parameters_for("chair")
    person = policy.parameters_for("person")

    assert wall.boundary_flux == pytest.approx(-1.0)
    assert chair.boundary_flux == pytest.approx(-3.0)
    assert person.boundary_flux == pytest.approx(-6.0)
    assert wall.tangential_strength == pytest.approx(0.0)
    assert chair.tangential_strength == pytest.approx(0.75)
    assert person.tangential_strength == pytest.approx(1.25)
    assert policy.known_classes == (
        "chair",
        "person",
        "unknown",
        "wall",
    )


def test_repository_configuration_loads() -> None:
    policy = SafeSAGEGuidancePolicy.from_json(
        PROJECT_ROOT / "configs" / "safe_sage_class_safety.json"
    )

    assert policy.parameters_for("chair").passing_side is PassingSide.RIGHT
    assert policy.parameters_for("person").boundary_flux == pytest.approx(
        -6.0
    )


def test_paper_baseline_configuration_loads_fixed_flux_values() -> None:
    policy = SafeSAGEGuidancePolicy.from_json(
        PROJECT_ROOT / "configs" / "safe_sage_paper_baseline.json"
    )

    static_object = policy.parameters_for("object")
    person = policy.parameters_for("person")

    assert static_object.boundary_flux == pytest.approx(-0.5)
    assert person.boundary_flux == pytest.approx(-1.7)
    assert static_object.tangential_strength == pytest.approx(1.25)
    assert person.tangential_strength == pytest.approx(1.25)
    assert static_object.passing_side is PassingSide.RIGHT
    assert person.passing_side is PassingSide.RIGHT


def test_lookup_normalizes_class_name(tmp_path) -> None:
    policy = SafeSAGEGuidancePolicy.from_json(
        write_config(tmp_path, default_config())
    )

    parameters = policy.parameters_for("  PERSON ")

    assert parameters.risk == pytest.approx(1.0)
    assert parameters.boundary_flux == pytest.approx(-6.0)


def test_unconfigured_class_uses_unknown_fallback(tmp_path) -> None:
    policy = SafeSAGEGuidancePolicy.from_json(
        write_config(tmp_path, default_config())
    )

    fallback = policy.parameters_for("fork_on_floor")

    assert fallback is policy.class_parameters["unknown"]
    assert fallback.risk == pytest.approx(1.0)
    assert fallback.boundary_flux == pytest.approx(-6.0)
    assert fallback.tangential_strength == pytest.approx(1.25)


def test_policy_accepts_semantic_obstacle(tmp_path) -> None:
    policy = SafeSAGEGuidancePolicy.from_json(
        write_config(tmp_path, default_config())
    )

    parameters = policy.parameters_for_obstacle(
        make_semantic_obstacle("chair")
    )

    assert parameters.risk == pytest.approx(0.4)
    assert parameters.tangential_strength == pytest.approx(0.75)


def test_policy_mapping_is_read_only(tmp_path) -> None:
    policy = SafeSAGEGuidancePolicy.from_json(
        write_config(tmp_path, default_config())
    )

    with pytest.raises(TypeError):
        policy.class_parameters["chair"] = make_parameters()


@pytest.mark.parametrize(
    "strength",
    [-0.01, float("inf"), float("nan")],
)
def test_invalid_tangential_strength_is_rejected(strength: float) -> None:
    with pytest.raises(ValueError):
        make_parameters(strength=strength)


@pytest.mark.parametrize("strength", [True, "strong", None])
def test_non_numeric_tangential_strength_is_rejected(
    strength: object,
) -> None:
    with pytest.raises(TypeError, match="real number"):
        make_parameters(strength=strength)


@pytest.mark.parametrize("side", ["clockwise", "", 3, None])
def test_invalid_passing_side_is_rejected(side: object) -> None:
    with pytest.raises((TypeError, ValueError), match="passing_side"):
        make_parameters(side=side)


def test_unknown_class_entry_is_required() -> None:
    with pytest.raises(ValueError, match="unknown_class"):
        SafeSAGEGuidancePolicy({"chair": make_parameters()})


def test_duplicate_normalized_class_is_rejected() -> None:
    parameters = make_parameters()

    with pytest.raises(ValueError, match="duplicate"):
        SafeSAGEGuidancePolicy(
            {
                "unknown": parameters,
                "Dining Table": parameters,
                "dining-table": parameters,
            }
        )


def test_wrong_schema_version_is_rejected(tmp_path) -> None:
    config = default_config()
    config["schema_version"] = 2

    with pytest.raises(ValueError, match="schema_version"):
        SafeSAGEGuidancePolicy.from_json(
            write_config(tmp_path, config)
        )


def test_missing_class_field_is_rejected(tmp_path) -> None:
    config = default_config()
    classes = config["classes"]
    classes["chair"] = {
        "risk": 0.4,
        "passing_side": "right",
    }

    with pytest.raises(ValueError, match="missing"):
        SafeSAGEGuidancePolicy.from_json(
            write_config(tmp_path, config)
        )


def test_unexpected_class_field_is_rejected(tmp_path) -> None:
    config = default_config()
    classes = config["classes"]
    classes["chair"]["signed_lambda"] = -0.75

    with pytest.raises(ValueError, match="unexpected"):
        SafeSAGEGuidancePolicy.from_json(
            write_config(tmp_path, config)
        )


def test_wrong_mapping_type_is_rejected(tmp_path) -> None:
    config = default_config()
    config["risk_flux_mapping"]["type"] = "quadratic"

    with pytest.raises(ValueError, match="linear"):
        SafeSAGEGuidancePolicy.from_json(
            write_config(tmp_path, config)
        )


def test_non_semantic_obstacle_is_rejected(tmp_path) -> None:
    policy = SafeSAGEGuidancePolicy.from_json(
        write_config(tmp_path, default_config())
    )

    with pytest.raises(TypeError, match="SemanticObstacle"):
        policy.parameters_for_obstacle("chair")
