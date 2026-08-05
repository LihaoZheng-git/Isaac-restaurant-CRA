from dataclasses import replace

import numpy as np
import pytest

from robot.safety.class_safety_policy import (
    ClassSafetyParameters,
    ClassSafetyPolicy,
)
from robot.safety.local_occupancy_grid import (
    LocalOccupancyGridConfig,
    build_local_occupancy_grid,
)
from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.poisson_safety_function import (
    PoissonSafetyFunctionConfig,
)
from robot.safety.safe_sage_configuration_space import (
    CircularRobotFootprint,
    RectangularRobotFootprint,
    SafeSAGEConfigurationSpaceConfig,
)
from robot.safety.safe_sage_field_pipeline import (
    SafeSAGEFieldPipelineConfig,
    SafeSAGEFieldPipelineError,
    build_safe_sage_field_pipeline,
)
from robot.safety.safe_sage_guidance_policy import (
    PassingSide,
    SafeSAGEGuidanceParameters,
    SafeSAGEGuidancePolicy,
)
from robot.safety.safe_sage_h_full import SafeSAGEHFullConfig
from robot.safety.semantic_obstacle import SemanticObstacle
from robot.safety.tangential_laplace_guidance_field import (
    TangentialLaplaceGuidanceFieldConfig,
)


def make_class_policy(
    *,
    chair_risk: float = 0.4,
    chair_flux: float = 3.0,
) -> ClassSafetyPolicy:
    """Return semantic parameters used to rasterize the source grid."""

    return ClassSafetyPolicy(
        {
            "unknown": ClassSafetyParameters(
                risk=1.0,
                boundary_flux_magnitude=6.0,
            ),
            "chair": ClassSafetyParameters(
                risk=chair_risk,
                boundary_flux_magnitude=chair_flux,
            ),
        }
    )


def make_guidance_policy(
    *,
    chair_risk: float = 0.4,
    chair_flux: float = 3.0,
    chair_tangential_strength: float = 0.75,
) -> SafeSAGEGuidancePolicy:
    """Return matching normal and tangential semantic parameters."""

    class_policy = make_class_policy(
        chair_risk=chair_risk,
        chair_flux=chair_flux,
    )
    return SafeSAGEGuidancePolicy(
        {
            "unknown": SafeSAGEGuidanceParameters(
                safety_parameters=class_policy.parameters_for("unknown"),
                tangential_strength=0.0,
                passing_side=PassingSide.RIGHT,
            ),
            "chair": SafeSAGEGuidanceParameters(
                safety_parameters=class_policy.parameters_for("chair"),
                tangential_strength=chair_tangential_strength,
                passing_side=PassingSide.LEFT,
            ),
        }
    )


def make_semantic_obstacle(track_id: int = 4) -> SemanticObstacle:
    """Return one deterministic chair track."""

    track = TrackedObstacle(
        track_id=track_id,
        centroid_world_xy=(0.0, 0.0),
        centroid_robot_xy=(0.0, 0.0),
        velocity_world_xy=(0.0, 0.0),
        velocity_robot_xy=(0.0, 0.0),
        point_count=1,
        observation_count=1,
        matched_previous_track=False,
    )
    return SemanticObstacle(
        tracked_obstacle=track,
        semantic_class="chair",
        semantic_confidence=1.0,
    )


def make_grid(
    *,
    obstacle_xy: tuple[float, float] = (12.5, 12.5),
    semantic: SemanticObstacle | None = None,
):
    """Return a large raw grid with one tracked obstacle point."""

    resolved_semantic = semantic or make_semantic_obstacle()
    return build_local_occupancy_grid(
        np.asarray([obstacle_xy], dtype=np.float64),
        point_track_ids=np.asarray(
            [resolved_semantic.track_id],
            dtype=np.int64,
        ),
        semantic_obstacles=(resolved_semantic,),
        class_safety_policy=make_class_policy(),
        config=LocalOccupancyGridConfig(
            resolution=1.0,
            minimum_x=0.0,
            maximum_x=25.0,
            minimum_y=0.0,
            maximum_y=25.0,
        ),
    )


def make_pipeline_config(
    *,
    yaw_slice_count: int = 3,
    footprint: (
        CircularRobotFootprint | RectangularRobotFootprint
    ) | None = None,
) -> SafeSAGEFieldPipelineConfig:
    """Return deterministic numerical settings for unit tests."""

    resolved_footprint = footprint or CircularRobotFootprint(radius=1.0)
    return SafeSAGEFieldPipelineConfig(
        configuration_space_config=(
            SafeSAGEConfigurationSpaceConfig(
                footprint=resolved_footprint,
                yaw_slice_count=yaw_slice_count,
            )
        ),
        guidance_field_config=(
            TangentialLaplaceGuidanceFieldConfig(
                convergence_tolerance=1e-7,
                maximum_iterations=20_000,
                residual_check_interval=5,
                relaxation_factor=1.7,
                tangential_buffer_radius=3.0,
            )
        ),
        h_full_config=SafeSAGEHFullConfig(
            yaw_slice_count=yaw_slice_count,
            poisson_config=PoissonSafetyFunctionConfig(
                convergence_tolerance=1e-7,
                maximum_iterations=20_000,
                residual_check_interval=5,
            ),
        ),
    )


def test_pipeline_builds_aligned_intermediate_and_full_fields() -> None:
    semantic = make_semantic_obstacle()
    result = build_safe_sage_field_pipeline(
        make_grid(semantic=semantic),
        guidance_policy=make_guidance_policy(),
        semantic_obstacles=(semantic,),
        config=make_pipeline_config(),
    )

    assert result.yaw_slice_count == 3
    assert result.semantic_obstacle_count == 1
    assert len(result.configuration_space.occupancy_grids) == 3
    assert len(result.guidance_fields) == 3
    assert result.h_full.shape == (3, 25, 25)
    np.testing.assert_array_equal(
        result.yaw_angles_radians,
        result.configuration_space.yaw_angles_radians,
    )

    for occupancy_grid, guidance_field in zip(
        result.configuration_space.occupancy_grids,
        result.guidance_fields,
        strict=True,
    ):
        assert occupancy_grid.shape == guidance_field.shape
        assert guidance_field.resolution == pytest.approx(
            occupancy_grid.config.resolution
        )


def test_circular_footprint_produces_zero_yaw_derivative() -> None:
    semantic = make_semantic_obstacle()
    result = build_safe_sage_field_pipeline(
        make_grid(semantic=semantic),
        guidance_policy=make_guidance_policy(),
        semantic_obstacles=(semantic,),
        config=make_pipeline_config(yaw_slice_count=4),
    )

    for yaw_index in range(1, result.yaw_slice_count):
        np.testing.assert_allclose(
            result.h_full.values[yaw_index],
            result.h_full.values[0],
            rtol=0.0,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        result.h_full.yaw_derivatives,
        0.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_rectangular_footprint_produces_yaw_dependent_geometry() -> None:
    semantic = make_semantic_obstacle()
    result = build_safe_sage_field_pipeline(
        make_grid(semantic=semantic),
        guidance_policy=make_guidance_policy(
            chair_tangential_strength=0.0
        ),
        semantic_obstacles=(semantic,),
        config=make_pipeline_config(
            yaw_slice_count=4,
            footprint=RectangularRobotFootprint.centered(
                length=4.0,
                width=2.0,
            ),
        ),
    )

    mask_at_zero = (
        result.configuration_space.occupancy_grids[2].occupied_mask
    )
    mask_at_half_pi = (
        result.configuration_space.occupancy_grids[3].occupied_mask
    )
    assert not np.array_equal(mask_at_zero, mask_at_half_pi)
    assert not np.array_equal(
        result.h_full.values[2],
        result.h_full.values[3],
    )


def test_semantic_tangential_guidance_is_preserved_in_every_slice() -> None:
    semantic = make_semantic_obstacle()
    result = build_safe_sage_field_pipeline(
        make_grid(semantic=semantic),
        guidance_policy=make_guidance_policy(
            chair_tangential_strength=0.75
        ),
        semantic_obstacles=(semantic,),
        config=make_pipeline_config(),
    )

    assert all(
        count > 0
        for count in result.tangential_interface_cell_counts
    )
    assert all(
        field.maximum_magnitude > 0.0
        for field in result.guidance_fields
    )


def test_semantic_generator_is_materialized_once_and_reused() -> None:
    semantic = make_semantic_obstacle()
    semantic_generator = (item for item in (semantic,))

    result = build_safe_sage_field_pipeline(
        make_grid(semantic=semantic),
        guidance_policy=make_guidance_policy(),
        semantic_obstacles=semantic_generator,
        config=make_pipeline_config(),
    )

    assert result.semantic_obstacle_count == 1
    assert all(
        field.tangential_interface_cell_count > 0
        for field in result.guidance_fields
    )


def test_mismatched_yaw_counts_are_rejected() -> None:
    config = make_pipeline_config()

    with pytest.raises(ValueError, match="yaw_slice_count"):
        replace(
            config,
            h_full_config=replace(
                config.h_full_config,
                yaw_slice_count=4,
            ),
        )


def test_grid_and_guidance_policy_metadata_mismatch_is_rejected() -> None:
    semantic = make_semantic_obstacle()

    with pytest.raises(ValueError, match="risk metadata"):
        build_safe_sage_field_pipeline(
            make_grid(semantic=semantic),
            guidance_policy=make_guidance_policy(chair_risk=0.8),
            semantic_obstacles=(semantic,),
            config=make_pipeline_config(),
        )


def test_guidance_failure_reports_yaw_slice_context() -> None:
    semantic = make_semantic_obstacle()

    with pytest.raises(
        SafeSAGEFieldPipelineError,
        match="yaw slice 0",
    ):
        build_safe_sage_field_pipeline(
            make_grid(
                obstacle_xy=(2.5, 12.5),
                semantic=semantic,
            ),
            guidance_policy=make_guidance_policy(),
            semantic_obstacles=(semantic,),
            config=make_pipeline_config(),
        )


def test_duplicate_semantic_tracks_are_rejected_before_solving() -> None:
    semantic = make_semantic_obstacle()

    with pytest.raises(ValueError, match="duplicate track ID"):
        build_safe_sage_field_pipeline(
            make_grid(semantic=semantic),
            guidance_policy=make_guidance_policy(),
            semantic_obstacles=(semantic, semantic),
            config=make_pipeline_config(),
        )


def test_result_diagnostics_are_consistent() -> None:
    semantic = make_semantic_obstacle()
    result = build_safe_sage_field_pipeline(
        make_grid(semantic=semantic),
        guidance_policy=make_guidance_policy(),
        semantic_obstacles=(semantic,),
        config=make_pipeline_config(),
    )

    assert len(result.guidance_iteration_counts) == 3
    assert all(count > 0 for count in result.guidance_iteration_counts)
    assert result.h_full.maximum_residuals.shape == (3,)
    assert np.all(result.h_full.maximum_residuals <= 1e-7)
    assert not result.yaw_angles_radians.flags.writeable
