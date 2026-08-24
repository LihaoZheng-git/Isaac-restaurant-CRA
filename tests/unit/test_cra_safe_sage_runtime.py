from dataclasses import replace

import numpy as np
import pytest

from robot.risk.cra import (
    PhysicalRiskFactor,
    PhysicalRiskProfile,
    RiskFactorEstimate,
)
from robot.safety.class_safety_policy import (
    ClassSafetyParameters,
    ClassSafetyPolicy,
)
from robot.safety.cra_safe_sage_policy import (
    CRASafeSAGEPolicy,
    CRASafeSAGEPolicyConfig,
)
from robot.safety.cra_track_policy import (
    TrackIdCRARiskProfileProvider,
)
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig
from robot.safety.obstacle_tracking import ObstacleTracker, ObstacleTrackerConfig
from robot.safety.poisson_safety_function import PoissonSafetyFunctionConfig
from robot.safety.safe_sage_configuration_space import (
    CircularRobotFootprint,
    SafeSAGEConfigurationSpaceConfig,
)
from robot.safety.safe_sage_field_pipeline import SafeSAGEFieldPipelineConfig
from robot.safety.safe_sage_guidance_policy import (
    PassingSide,
    SafeSAGEGuidanceParameters,
    SafeSAGEGuidancePolicy,
)
from robot.safety.safe_sage_h_full import SafeSAGEHFullConfig
from robot.safety.safe_sage_runtime import (
    SafeSAGEPerceptionObservation,
    SafeSAGEPerceptionRuntime,
)
from robot.safety.semantic_label_provider import (
    SemanticClassLabel,
    TrackIdSemanticLabelProvider,
)
from robot.safety.tangential_laplace_guidance_field import (
    TangentialLaplaceGuidanceFieldConfig,
)


TRACK_ZERO_POINTS = np.array(
    [
        [1.9, -2.1],
        [1.9, -1.9],
        [2.1, -2.1],
        [2.1, -1.9],
    ],
    dtype=np.float64,
)
TRACK_ONE_POINTS = np.array(
    [
        [1.9, 1.9],
        [1.9, 2.1],
        [2.1, 1.9],
        [2.1, 2.1],
    ],
    dtype=np.float64,
)
TWO_TRACK_POINTS = np.vstack((TRACK_ZERO_POINTS, TRACK_ONE_POINTS))


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


def make_baseline_policies() -> tuple[ClassSafetyPolicy, SafeSAGEGuidancePolicy]:
    """Return a same-class baseline that CRA can override per track."""

    class_policy = ClassSafetyPolicy(
        {
            "unknown": ClassSafetyParameters(
                risk=1.0,
                boundary_flux_magnitude=1.7,
            ),
            "object": ClassSafetyParameters(
                risk=0.0,
                boundary_flux_magnitude=0.5,
            ),
        }
    )
    guidance_policy = SafeSAGEGuidancePolicy(
        {
            "unknown": SafeSAGEGuidanceParameters(
                safety_parameters=class_policy.parameters_for("unknown"),
                tangential_strength=0.0,
                passing_side=PassingSide.RIGHT,
            ),
            "object": SafeSAGEGuidanceParameters(
                safety_parameters=class_policy.parameters_for("object"),
                tangential_strength=0.0,
                passing_side=PassingSide.RIGHT,
            ),
        }
    )
    return class_policy, guidance_policy


def make_runtime(
    *,
    with_cra_policy: bool = True,
    risk_dependent_tangent: bool = False,
) -> SafeSAGEPerceptionRuntime:
    """Return a small deterministic two-track numerical runtime."""

    class_policy, guidance_policy = make_baseline_policies()
    cra_policy = (
        CRASafeSAGEPolicy(
            config=CRASafeSAGEPolicyConfig(
                tangential_strength_at_zero_risk=(
                    0.2 if risk_dependent_tangent else 0.0
                ),
                tangential_strength_at_unit_risk=(
                    1.2 if risk_dependent_tangent else 0.0
                ),
            )
        )
        if with_cra_policy
        else None
    )
    return SafeSAGEPerceptionRuntime(
        obstacle_tracker=ObstacleTracker(
            ObstacleTrackerConfig(
                cluster_radius=0.5,
                minimum_cluster_points=2,
                maximum_association_distance=0.8,
                velocity_smoothing_factor=1.0,
            )
        ),
        class_policy=class_policy,
        guidance_policy=guidance_policy,
        grid_config=LocalOccupancyGridConfig(
            resolution=1.0,
            minimum_x=-5.0,
            maximum_x=8.0,
            minimum_y=-6.0,
            maximum_y=6.0,
        ),
        field_config=SafeSAGEFieldPipelineConfig(
            configuration_space_config=SafeSAGEConfigurationSpaceConfig(
                footprint=CircularRobotFootprint(radius=0.5),
                yaw_slice_count=3,
            ),
            guidance_field_config=TangentialLaplaceGuidanceFieldConfig(
                convergence_tolerance=1e-6,
                maximum_iterations=10_000,
                residual_check_interval=5,
                tangential_buffer_radius=2.0,
                allow_dense_scene_arbitration=risk_dependent_tangent,
            ),
            h_full_config=SafeSAGEHFullConfig(
                yaw_slice_count=3,
                poisson_config=PoissonSafetyFunctionConfig(
                    convergence_tolerance=1e-6,
                    maximum_iterations=10_000,
                    residual_check_interval=5,
                ),
            ),
        ),
        enable_temporal_updates=False,
        cra_policy=cra_policy,
    )


def make_observation(
    frame_id: int = 1,
    timestamp: float = 1.0,
) -> SafeSAGEPerceptionObservation:
    """Return two separated same-geometry obstacle clusters."""

    return SafeSAGEPerceptionObservation(
        sensor_frame_id=frame_id,
        sensor_timestamp_seconds=timestamp,
        simulation_time_seconds=timestamp,
        obstacle_points_robot_xy=TWO_TRACK_POINTS,
        robot_position_world_xy=np.zeros(2),
        robot_yaw=0.0,
    )


def make_label_provider() -> TrackIdSemanticLabelProvider:
    """Assign the same baseline semantic class to both tracks."""

    label = SemanticClassLabel(
        "object",
        confidence=1.0,
        source="controlled_oracle",
    )
    return TrackIdSemanticLabelProvider({0: label, 1: label})


def make_risk_provider(
    first_risk: float = 0.1,
    second_risk: float = 0.9,
) -> TrackIdCRARiskProfileProvider:
    """Return different contextual profiles for the two same-class tracks."""

    return TrackIdCRARiskProfileProvider(
        {
            0: profile_with_risk(first_risk),
            1: profile_with_risk(
                second_risk,
                factor=PhysicalRiskFactor.CONTACT,
            ),
        }
    )


def assert_grid_track_policy(
    runtime: SafeSAGEPerceptionRuntime,
    *,
    track_id: int,
    risk: float,
) -> None:
    """Assert all configuration cells owned by a track carry CRA metadata."""

    grid = runtime.field.pipeline.configuration_space.occupancy_grids[0]
    track_mask = grid.occupied_mask & (grid.cell_track_ids == track_id)
    assert np.any(track_mask)
    np.testing.assert_allclose(grid.cell_risks[track_mask], risk, atol=1e-12)
    expected_flux = -(0.5 + risk * (1.7 - 0.5))
    np.testing.assert_allclose(
        grid.cell_boundary_fluxes[track_mask],
        expected_flux,
        atol=1e-12,
    )


def test_runtime_applies_different_cra_policy_to_same_class_tracks() -> None:
    runtime = make_runtime()

    status = runtime.initialize_with_cra(
        make_observation(),
        make_label_provider(),
        make_risk_provider(),
    )
    field = status.snapshot.payload

    assert field.has_cra_policy
    assert field.selected_track_ids == (0, 1)
    assert field.cra_track_ids == (0, 1)
    assert field.cra_fallback_track_ids == ()
    assert all(
        obstacle.semantic_class == "object"
        for obstacle in field.label_result.semantic_obstacles
    )
    assert_grid_track_policy(runtime, track_id=0, risk=0.1)
    assert_grid_track_policy(runtime, track_id=1, risk=0.9)


def test_runtime_applies_cra_tangential_strength_to_harmonic_boundaries() -> None:
    runtime = make_runtime(risk_dependent_tangent=True)

    runtime.initialize_with_cra(
        make_observation(),
        make_label_provider(),
        make_risk_provider(0.1, 0.9),
    )
    guidance = runtime.field.pipeline.guidance_fields[0]
    boundary_magnitudes = np.linalg.norm(
        guidance.tangential_boundary_vectors_robot_xy[
            guidance.tangential_interface_mask
        ],
        axis=1,
    )

    assert guidance.tangential_interface_cell_count > 0
    assert np.any(np.isclose(boundary_magnitudes, 0.3, atol=1e-12))
    assert np.any(np.isclose(boundary_magnitudes, 1.1, atol=1e-12))


def test_missing_track_profile_fails_closed_inside_runtime() -> None:
    runtime = make_runtime()
    provider = TrackIdCRARiskProfileProvider({0: profile_with_risk(0.1)})

    runtime.initialize_with_cra(
        make_observation(),
        make_label_provider(),
        provider,
    )

    assert runtime.field.cra_fallback_track_ids == (1,)
    assert runtime.field.cra_policy_result.assignment_for(
        1
    ).policy_result.risk == pytest.approx(1.0)
    assert_grid_track_policy(runtime, track_id=1, risk=1.0)


def test_semantic_only_entry_point_remains_unchanged_with_cra_configured() -> None:
    runtime = make_runtime()

    runtime.initialize(make_observation(), make_label_provider())

    assert not runtime.field.has_cra_policy
    assert runtime.field.cra_track_ids == ()
    assert runtime.field.cra_fallback_track_ids == ()
    assert_grid_track_policy(runtime, track_id=0, risk=0.0)
    assert_grid_track_policy(runtime, track_id=1, risk=0.0)


def test_cra_entry_point_requires_configured_policy_without_mutation() -> None:
    runtime = make_runtime(with_cra_policy=False)

    with pytest.raises(RuntimeError, match="Configure cra_policy"):
        runtime.initialize_with_cra(
            make_observation(),
            make_label_provider(),
            make_risk_provider(),
        )

    assert not runtime.initialized
    assert runtime.active_track_count == 0


def test_contextual_risk_can_change_on_new_accepted_frame() -> None:
    runtime = make_runtime()
    labels = make_label_provider()
    runtime.initialize_with_cra(
        make_observation(1, 1.0),
        labels,
        make_risk_provider(0.1, 0.9),
    )

    status = runtime.update_with_cra(
        make_observation(2, 1.2),
        labels,
        make_risk_provider(0.8, 0.4),
    )

    assert status.generation == 2
    assert status.snapshot.payload.cra_policy_result.assignment_for(
        0
    ).policy_result.risk == pytest.approx(0.8)
    assert status.snapshot.payload.cra_policy_result.assignment_for(
        1
    ).policy_result.risk == pytest.approx(0.4)
    assert_grid_track_policy(runtime, track_id=0, risk=0.8)
    assert_grid_track_policy(runtime, track_id=1, risk=0.4)


class InvalidRiskProfileProvider:
    """Protocol-shaped provider returning an invalid candidate value."""

    def profile_for(self, obstacle):
        del obstacle
        return "risk"


def test_failed_cra_update_preserves_field_and_tracker_transaction() -> None:
    runtime = make_runtime()
    labels = make_label_provider()
    runtime.initialize_with_cra(
        make_observation(1, 1.0),
        labels,
        make_risk_provider(),
    )
    original_field = runtime.field

    with pytest.raises(TypeError, match="PhysicalRiskProfile or None"):
        runtime.update_with_cra(
            make_observation(2, 1.2),
            labels,
            InvalidRiskProfileProvider(),
        )

    assert runtime.field is original_field
    assert runtime.field_buffer.generation == 1
    assert runtime.active_track_count == 2


def test_runtime_rejects_non_protocol_profile_provider() -> None:
    runtime = make_runtime()

    with pytest.raises(TypeError, match="CRARiskProfileProvider"):
        runtime.initialize_with_cra(
            make_observation(),
            make_label_provider(),
            object(),
        )

    assert not runtime.initialized


def test_runtime_constructor_rejects_invalid_cra_policy() -> None:
    class_policy, guidance_policy = make_baseline_policies()

    with pytest.raises(TypeError, match="cra_policy"):
        SafeSAGEPerceptionRuntime(
            obstacle_tracker=ObstacleTracker(),
            class_policy=class_policy,
            guidance_policy=guidance_policy,
            grid_config=LocalOccupancyGridConfig(),
            field_config=SafeSAGEFieldPipelineConfig(
                configuration_space_config=(
                    SafeSAGEConfigurationSpaceConfig(
                        footprint=CircularRobotFootprint(radius=0.5),
                        yaw_slice_count=3,
                    )
                ),
                h_full_config=SafeSAGEHFullConfig(yaw_slice_count=3),
            ),
            cra_policy="policy",
        )


def test_runtime_field_rejects_invalid_cra_diagnostics() -> None:
    runtime = make_runtime()
    runtime.initialize_with_cra(
        make_observation(),
        make_label_provider(),
        make_risk_provider(),
    )

    with pytest.raises(TypeError, match="cra_policy_result"):
        replace(runtime.field, cra_policy_result="risk")
