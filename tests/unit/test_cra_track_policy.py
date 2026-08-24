import json

import pytest

from robot.risk.cra import (
    PhysicalRiskFactor,
    PhysicalRiskProfile,
    RiskFactorEstimate,
)
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy
from robot.safety.cra_track_policy import (
    CRARiskProfileProvider,
    CRATrackPolicyAssignment,
    CRATrackPolicyResult,
    TrackIdCRARiskProfileProvider,
    resolve_cra_track_policy,
)
from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.semantic_obstacle import SemanticObstacle


def profile_with_risk(
    risk: float,
    *,
    factor: PhysicalRiskFactor = PhysicalRiskFactor.THERMAL,
) -> PhysicalRiskProfile:
    """Return one controlled profile with exactly one active factor."""

    estimates = {
        item.value: RiskFactorEstimate.known_absent(source="test")
        for item in PhysicalRiskFactor
    }
    estimates[factor.value] = RiskFactorEstimate(
        severity=risk,
        interaction_likelihood=1.0,
        confidence=1.0,
        source="test",
    )
    return PhysicalRiskProfile(**estimates)


def semantic_obstacle(
    track_id: int,
    *,
    semantic_class: str = "object",
) -> SemanticObstacle:
    """Return one deterministic semantic track."""

    return SemanticObstacle(
        tracked_obstacle=TrackedObstacle(
            track_id=track_id,
            centroid_world_xy=(2.0, float(track_id)),
            centroid_robot_xy=(2.0, float(track_id)),
            velocity_world_xy=(0.0, 0.0),
            velocity_robot_xy=(0.0, 0.0),
            point_count=4,
            observation_count=1,
            matched_previous_track=False,
        ),
        semantic_class=semantic_class,
        semantic_confidence=0.9,
        label_source="test",
    )


def test_track_id_provider_implements_runtime_protocol() -> None:
    provider = TrackIdCRARiskProfileProvider({0: profile_with_risk(0.2)})

    assert isinstance(provider, CRARiskProfileProvider)


def test_track_id_provider_returns_matching_profile_or_none() -> None:
    profile = profile_with_risk(0.2)
    provider = TrackIdCRARiskProfileProvider({2: profile})

    assert provider.profile_for(semantic_obstacle(2)) is profile
    assert provider.profile_for(semantic_obstacle(3)) is None


def test_track_id_provider_mapping_is_read_only() -> None:
    provider = TrackIdCRARiskProfileProvider({0: profile_with_risk(0.2)})

    with pytest.raises(TypeError):
        provider.profiles_by_track_id[1] = profile_with_risk(0.4)


@pytest.mark.parametrize("track_id", [-1, True, "zero"])
def test_invalid_provider_track_id_is_rejected(track_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        TrackIdCRARiskProfileProvider({track_id: profile_with_risk(0.2)})


def test_invalid_provider_profile_is_rejected() -> None:
    with pytest.raises(TypeError, match="PhysicalRiskProfile"):
        TrackIdCRARiskProfileProvider({0: "risk"})


def test_provider_rejects_non_semantic_obstacle() -> None:
    provider = TrackIdCRARiskProfileProvider({})

    with pytest.raises(TypeError, match="SemanticObstacle"):
        provider.profile_for("object")


def test_resolution_is_sorted_by_geometric_track_id() -> None:
    obstacles = (semantic_obstacle(4), semantic_obstacle(1))
    result = resolve_cra_track_policy(
        obstacles,
        profile_provider=TrackIdCRARiskProfileProvider(
            {
                1: profile_with_risk(0.2),
                4: profile_with_risk(0.8),
            }
        ),
        policy=CRASafeSAGEPolicy(),
    )

    assert result.track_ids == (1, 4)
    assert result.assignment_for(1).policy_result.risk == pytest.approx(0.2)
    assert result.assignment_for(4).policy_result.risk == pytest.approx(0.8)


def test_missing_profile_uses_conservative_unknown_fallback() -> None:
    result = resolve_cra_track_policy(
        (semantic_obstacle(3),),
        profile_provider=TrackIdCRARiskProfileProvider({}),
        policy=CRASafeSAGEPolicy(),
    )
    assignment = result.assignment_for(3)

    assert assignment.used_fallback_profile
    assert assignment.policy_result.risk == pytest.approx(1.0)
    assert assignment.policy_result.boundary_flux == pytest.approx(-1.7)
    assert result.fallback_track_ids == (3,)


def test_explicit_profile_is_not_marked_as_fallback() -> None:
    result = resolve_cra_track_policy(
        (semantic_obstacle(3),),
        profile_provider=TrackIdCRARiskProfileProvider({3: profile_with_risk(0.4)}),
        policy=CRASafeSAGEPolicy(),
    )

    assert not result.assignment_for(3).used_fallback_profile
    assert result.fallback_track_ids == ()


def test_same_semantic_class_can_receive_different_contextual_policy() -> None:
    result = resolve_cra_track_policy(
        (semantic_obstacle(0), semantic_obstacle(1)),
        profile_provider=TrackIdCRARiskProfileProvider(
            {
                0: profile_with_risk(0.1),
                1: profile_with_risk(0.9),
            }
        ),
        policy=CRASafeSAGEPolicy(),
    )

    low = result.assignment_for(0)
    high = result.assignment_for(1)
    assert low.obstacle.semantic_class == high.obstacle.semantic_class
    assert low.policy_result.risk == pytest.approx(0.1)
    assert high.policy_result.risk == pytest.approx(0.9)
    assert abs(high.policy_result.boundary_flux) > abs(low.policy_result.boundary_flux)


def test_parameter_mappings_are_immutable_and_consistent() -> None:
    result = resolve_cra_track_policy(
        (semantic_obstacle(2),),
        profile_provider=TrackIdCRARiskProfileProvider({2: profile_with_risk(0.6)}),
        policy=CRASafeSAGEPolicy(),
    )

    assert result.safety_parameters_by_track_id[2].risk == pytest.approx(0.6)
    assert result.guidance_parameters_by_track_id[2].risk == pytest.approx(0.6)
    with pytest.raises(TypeError):
        result.safety_parameters_by_track_id[3] = result.assignment_for(
            2
        ).safety_parameters
    with pytest.raises(TypeError):
        result.guidance_parameters_by_track_id[3] = result.assignment_for(
            2
        ).guidance_parameters


def test_result_is_json_serializable() -> None:
    result = resolve_cra_track_policy(
        (semantic_obstacle(2, semantic_class="knife"),),
        profile_provider=TrackIdCRARiskProfileProvider(
            {
                2: profile_with_risk(
                    0.8,
                    factor=PhysicalRiskFactor.CONTACT,
                )
            }
        ),
        policy=CRASafeSAGEPolicy(),
    )
    payload = result.as_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["track_ids"] == [2]
    assert payload["assignments"][0]["semantic_class"] == "knife"
    assert '"dominant_factor": "contact"' in encoded


def test_duplicate_semantic_track_is_rejected() -> None:
    obstacle = semantic_obstacle(1)

    with pytest.raises(ValueError, match="duplicate track ID"):
        resolve_cra_track_policy(
            (obstacle, obstacle),
            profile_provider=TrackIdCRARiskProfileProvider({}),
            policy=CRASafeSAGEPolicy(),
        )


class InvalidProfileProvider:
    """Protocol-shaped provider returning an invalid profile."""

    def profile_for(self, obstacle: SemanticObstacle):
        del obstacle
        return "risk"


def test_invalid_profile_provider_result_is_rejected() -> None:
    with pytest.raises(TypeError, match="PhysicalRiskProfile or None"):
        resolve_cra_track_policy(
            (semantic_obstacle(0),),
            profile_provider=InvalidProfileProvider(),
            policy=CRASafeSAGEPolicy(),
        )


def test_invalid_resolution_dependencies_and_obstacles_are_rejected() -> None:
    provider = TrackIdCRARiskProfileProvider({})
    policy = CRASafeSAGEPolicy()

    with pytest.raises(TypeError, match="profile_provider"):
        resolve_cra_track_policy((), profile_provider=object(), policy=policy)
    with pytest.raises(TypeError, match="policy"):
        resolve_cra_track_policy((), profile_provider=provider, policy=object())
    with pytest.raises(TypeError, match="semantic_obstacles"):
        resolve_cra_track_policy(None, profile_provider=provider, policy=policy)
    with pytest.raises(TypeError, match="SemanticObstacle"):
        resolve_cra_track_policy(
            ("object",),
            profile_provider=provider,
            policy=policy,
        )


def test_assignment_and_result_validate_members() -> None:
    obstacle = semantic_obstacle(0)
    profile = profile_with_risk(0.4)
    policy_result = CRASafeSAGEPolicy().evaluate(profile)
    assignment = CRATrackPolicyAssignment(
        obstacle=obstacle,
        profile=profile,
        policy_result=policy_result,
    )

    with pytest.raises(TypeError, match="obstacle"):
        CRATrackPolicyAssignment("object", profile, policy_result)
    with pytest.raises(TypeError, match="profile"):
        CRATrackPolicyAssignment(obstacle, "risk", policy_result)
    with pytest.raises(TypeError, match="policy_result"):
        CRATrackPolicyAssignment(obstacle, profile, "policy")
    with pytest.raises(TypeError, match="boolean"):
        CRATrackPolicyAssignment(obstacle, profile, policy_result, 1)
    with pytest.raises(TypeError, match="CRATrackPolicyAssignment"):
        CRATrackPolicyResult(("assignment",))
    with pytest.raises(ValueError, match="duplicate"):
        CRATrackPolicyResult((assignment, assignment))


def test_assignment_lookup_validates_track_id() -> None:
    result = resolve_cra_track_policy(
        (semantic_obstacle(2),),
        profile_provider=TrackIdCRARiskProfileProvider({}),
        policy=CRASafeSAGEPolicy(),
    )

    with pytest.raises(TypeError, match="integer"):
        result.assignment_for(True)
    with pytest.raises(KeyError, match="track ID 9"):
        result.assignment_for(9)
