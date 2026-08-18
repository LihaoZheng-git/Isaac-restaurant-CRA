from dataclasses import replace

import numpy as np
import pytest

from robot.safety.class_safety_policy import (
    ClassSafetyParameters,
    ClassSafetyPolicy,
)
from robot.safety.laplace_guidance_field import (
    LaplaceGuidanceFieldConfig,
    LaplaceGuidanceFieldResult,
    solve_laplace_guidance_field,
)
from robot.safety.local_occupancy_grid import (
    LocalOccupancyGridConfig,
    build_local_occupancy_grid,
)
from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.safe_sage_guidance_policy import (
    PassingSide,
    SafeSAGEGuidanceParameters,
    SafeSAGEGuidancePolicy,
)
from robot.safety.semantic_obstacle import SemanticObstacle
from robot.safety.tangential_laplace_guidance_field import (
    TangentialLaplaceGuidanceFieldConfig,
    solve_tangential_laplace_guidance_field,
)


def make_safety_parameters(
    risk: float,
    flux_magnitude: float,
) -> ClassSafetyParameters:
    """Return one normal-flux parameter object."""

    return ClassSafetyParameters(
        risk=risk,
        boundary_flux_magnitude=flux_magnitude,
    )


def make_class_policy() -> ClassSafetyPolicy:
    """Return a normal-flux policy matching the guidance policy."""

    return ClassSafetyPolicy(
        {
            "unknown": make_safety_parameters(1.0, 6.0),
            "wall": make_safety_parameters(0.0, 1.0),
            "chair": make_safety_parameters(0.4, 3.0),
            "person": make_safety_parameters(1.0, 6.0),
        }
    )


def make_guidance_policy(
    *,
    chair_strength: float = 0.75,
    chair_side: PassingSide = PassingSide.LEFT,
    person_strength: float = 1.25,
    person_side: PassingSide = PassingSide.RIGHT,
    unknown_strength: float = 1.25,
) -> SafeSAGEGuidancePolicy:
    """Return a deterministic normal-and-tangential policy."""

    class_policy = make_class_policy()

    def parameters(
        semantic_class: str,
        strength: float,
        side: PassingSide,
    ) -> SafeSAGEGuidanceParameters:
        return SafeSAGEGuidanceParameters(
            safety_parameters=class_policy.parameters_for(semantic_class),
            tangential_strength=strength,
            passing_side=side,
        )

    return SafeSAGEGuidancePolicy(
        {
            "unknown": parameters(
                "unknown",
                unknown_strength,
                PassingSide.RIGHT,
            ),
            "wall": parameters("wall", 0.0, PassingSide.RIGHT),
            "chair": parameters(
                "chair",
                chair_strength,
                chair_side,
            ),
            "person": parameters(
                "person",
                person_strength,
                person_side,
            ),
        }
    )


def make_semantic_obstacle(
    track_id: int,
    semantic_class: str,
) -> SemanticObstacle:
    """Return one deterministic semantic obstacle."""

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
        semantic_class=semantic_class,
        semantic_confidence=1.0,
    )


def make_grid(
    *,
    size: int = 17,
    points: np.ndarray | None = None,
    track_ids: np.ndarray | None = None,
    semantics: tuple[SemanticObstacle, ...] = (),
):
    """Return a square unit-resolution occupancy grid."""

    obstacle_points = (
        np.empty((0, 2), dtype=np.float64)
        if points is None
        else points
    )
    return build_local_occupancy_grid(
        obstacle_points,
        point_track_ids=track_ids,
        semantic_obstacles=semantics,
        class_safety_policy=make_class_policy(),
        config=LocalOccupancyGridConfig(
            resolution=1.0,
            minimum_x=0.0,
            maximum_x=float(size),
            minimum_y=0.0,
            maximum_y=float(size),
        ),
    )


def make_config(
    *,
    buffer_radius: float = 3.0,
) -> TangentialLaplaceGuidanceFieldConfig:
    """Return deterministic numerical settings for unit tests."""

    return TangentialLaplaceGuidanceFieldConfig(
        convergence_tolerance=1e-8,
        maximum_iterations=20_000,
        residual_check_interval=5,
        relaxation_factor=1.7,
        tangential_buffer_radius=buffer_radius,
    )


def solve_single_obstacle(
    *,
    policy: SafeSAGEGuidancePolicy | None = None,
    semantic_class: str = "chair",
):
    """Solve one centered tracked-obstacle field."""

    semantic = make_semantic_obstacle(4, semantic_class)
    grid = make_grid(
        points=np.array([[8.5, 8.5]]),
        track_ids=np.array([4]),
        semantics=(semantic,),
    )
    return solve_tangential_laplace_guidance_field(
        grid,
        guidance_policy=policy or make_guidance_policy(),
        semantic_obstacles=(semantic,),
        config=make_config(),
    )


def test_empty_grid_produces_zero_guidance() -> None:
    result = solve_tangential_laplace_guidance_field(
        make_grid(),
        guidance_policy=make_guidance_policy(),
        config=make_config(),
    )

    assert result.iteration_count == 1
    assert result.maximum_residual == pytest.approx(0.0)
    assert result.maximum_magnitude == pytest.approx(0.0)
    assert result.tangential_interface_cell_count == 0
    assert not np.any(result.obstacle_interface_mask)


def test_result_remains_compatible_with_existing_filter_contract() -> None:
    result = solve_single_obstacle()

    assert isinstance(result, LaplaceGuidanceFieldResult)
    assert result.shape == (17, 17)
    assert result.maximum_residual <= 1e-8


def test_normal_boundary_condition_is_preserved() -> None:
    result = solve_single_obstacle()

    assert result.vector_at_cell(8, 7) == pytest.approx((-3.0, 0.0))
    assert result.obstacle_normals_robot_xy[8, 7] == pytest.approx(
        (1.0, 0.0)
    )


def test_pass_left_points_toward_positive_robot_y() -> None:
    result = solve_single_obstacle(
        policy=make_guidance_policy(
            chair_side=PassingSide.LEFT,
        )
    )

    assert result.tangential_interface_mask[8, 5]
    assert result.tangential_normals_robot_xy[8, 5] == pytest.approx(
        (1.0, 0.0)
    )
    assert result.vector_at_cell(8, 5) == pytest.approx((0.0, 0.75))


def test_pass_right_points_toward_negative_robot_y() -> None:
    result = solve_single_obstacle(
        policy=make_guidance_policy(
            chair_side=PassingSide.RIGHT,
        )
    )

    assert result.vector_at_cell(8, 5) == pytest.approx((0.0, -0.75))


def test_left_and_right_fields_reverse_tangential_direction() -> None:
    left = solve_single_obstacle(
        policy=make_guidance_policy(chair_side=PassingSide.LEFT)
    )
    right = solve_single_obstacle(
        policy=make_guidance_policy(chair_side=PassingSide.RIGHT)
    )

    np.testing.assert_allclose(
        left.tangential_boundary_vectors_robot_xy,
        -right.tangential_boundary_vectors_robot_xy,
        atol=1e-12,
    )


def test_tangential_guidance_propagates_into_free_space() -> None:
    left = solve_single_obstacle(
        policy=make_guidance_policy(chair_side=PassingSide.LEFT)
    )
    right = solve_single_obstacle(
        policy=make_guidance_policy(chair_side=PassingSide.RIGHT)
    )

    # This point lies two cells outside the fixed tangential interface.
    # Its nonzero lateral component therefore comes from the harmonic solve,
    # not directly from a prescribed boundary value.
    assert not left.dirichlet_boundary_mask[8, 3]
    left_vector = left.vector_at_cell(8, 3)
    right_vector = right.vector_at_cell(8, 3)

    assert left_vector[1] > 0.0
    assert right_vector[1] < 0.0
    assert left_vector[0] == pytest.approx(right_vector[0], abs=1e-12)
    assert left_vector[1] == pytest.approx(-right_vector[1], abs=1e-12)


def test_tangential_boundary_is_perpendicular_with_requested_strength() -> None:
    result = solve_single_obstacle(
        policy=make_guidance_policy(chair_strength=0.75)
    )
    mask = result.tangential_interface_mask
    normals = result.tangential_normals_robot_xy[mask]
    vectors = result.tangential_boundary_vectors_robot_xy[mask]

    dot_products = np.sum(normals * vectors, axis=1)
    magnitudes = np.linalg.norm(vectors, axis=1)

    np.testing.assert_allclose(dot_products, 0.0, atol=1e-12)
    np.testing.assert_allclose(magnitudes, 0.75, atol=1e-12)


def test_larger_strength_increases_tangential_magnitude() -> None:
    weak = solve_single_obstacle(
        policy=make_guidance_policy(chair_strength=0.5)
    )
    strong = solve_single_obstacle(
        policy=make_guidance_policy(chair_strength=1.5)
    )

    weak_magnitude = np.linalg.norm(weak.vector_at_cell(8, 5))
    strong_magnitude = np.linalg.norm(strong.vector_at_cell(8, 5))

    assert weak_magnitude == pytest.approx(0.5)
    assert strong_magnitude == pytest.approx(1.5)


def test_classes_prescribe_independent_normal_and_tangential_strengths() -> None:
    chair = make_semantic_obstacle(2, "chair")
    person = make_semantic_obstacle(8, "person")
    grid = make_grid(
        size=29,
        points=np.array([[8.5, 14.5], [20.5, 14.5]]),
        track_ids=np.array([2, 8]),
        semantics=(chair, person),
    )
    result = solve_tangential_laplace_guidance_field(
        grid,
        guidance_policy=make_guidance_policy(),
        semantic_obstacles=(chair, person),
        config=make_config(),
    )

    # The two obstacles have identical one-cell geometry. Their normal and
    # tangential boundary magnitudes differ only because their classes do.
    assert np.linalg.norm(result.vector_at_cell(14, 7)) == pytest.approx(
        3.0
    )
    assert np.linalg.norm(result.vector_at_cell(14, 19)) == pytest.approx(
        6.0
    )
    assert np.linalg.norm(result.vector_at_cell(14, 5)) == pytest.approx(
        0.75
    )
    assert np.linalg.norm(result.vector_at_cell(14, 17)) == pytest.approx(
        1.25
    )


def test_zero_strength_recovers_normal_only_solution() -> None:
    semantic = make_semantic_obstacle(4, "chair")
    grid = make_grid(
        points=np.array([[8.5, 8.5]]),
        track_ids=np.array([4]),
        semantics=(semantic,),
    )
    normal_config = LaplaceGuidanceFieldConfig(
        convergence_tolerance=1e-8,
        maximum_iterations=20_000,
        residual_check_interval=5,
        relaxation_factor=1.7,
    )
    normal_only = solve_laplace_guidance_field(grid, normal_config)
    tangential_ablation = solve_tangential_laplace_guidance_field(
        grid,
        guidance_policy=make_guidance_policy(chair_strength=0.0),
        semantic_obstacles=(semantic,),
        config=make_config(),
    )

    assert tangential_ablation.tangential_interface_cell_count == 0
    np.testing.assert_allclose(
        tangential_ablation.vectors_robot_xy,
        normal_only.vectors_robot_xy,
        rtol=0.0,
        atol=1e-12,
    )


def test_repeated_solve_is_exactly_deterministic() -> None:
    first = solve_single_obstacle()
    second = solve_single_obstacle()

    np.testing.assert_array_equal(
        first.vectors_robot_xy,
        second.vectors_robot_xy,
    )
    assert first.iteration_count == second.iteration_count
    assert first.maximum_residual == second.maximum_residual


def test_unlabeled_track_uses_unknown_parameters() -> None:
    grid = make_grid(
        points=np.array([[8.5, 8.5]]),
        track_ids=np.array([9]),
    )
    result = solve_tangential_laplace_guidance_field(
        grid,
        guidance_policy=make_guidance_policy(unknown_strength=1.25),
        config=make_config(),
    )

    assert result.vector_at_cell(8, 7) == pytest.approx((-6.0, 0.0))
    assert np.linalg.norm(result.vector_at_cell(8, 5)) == pytest.approx(
        1.25
    )


def test_overlapping_buffers_are_rejected() -> None:
    first = make_semantic_obstacle(2, "chair")
    second = make_semantic_obstacle(8, "person")
    grid = make_grid(
        size=21,
        points=np.array([[7.5, 10.5], [12.5, 10.5]]),
        track_ids=np.array([2, 8]),
        semantics=(first, second),
    )

    with pytest.raises(ValueError, match="buffers overlap"):
        solve_tangential_laplace_guidance_field(
            grid,
            guidance_policy=make_guidance_policy(),
            semantic_obstacles=(first, second),
            config=make_config(),
        )


def test_dense_scene_arbitration_resolves_overlapping_buffers() -> None:
    first = make_semantic_obstacle(2, "chair")
    second = make_semantic_obstacle(8, "person")
    grid = make_grid(
        size=21,
        points=np.array([[7.5, 10.5], [12.5, 10.5]]),
        track_ids=np.array([2, 8]),
        semantics=(first, second),
    )

    result = solve_tangential_laplace_guidance_field(
        grid,
        guidance_policy=make_guidance_policy(),
        semantic_obstacles=(first, second),
        config=TangentialLaplaceGuidanceFieldConfig(
            tangential_buffer_radius=3.0,
            allow_dense_scene_arbitration=True,
            convergence_tolerance=1e-6,
            maximum_iterations=10_000,
            residual_check_interval=5,
        ),
    )

    assert result.tangential_interface_cell_count > 0
    assert np.all(np.isfinite(result.vectors_robot_xy))
    assert not np.any(
        result.tangential_interface_mask
        & result.obstacle_interface_mask
    )


def test_dense_scene_arbitration_accepts_symmetric_normal_saddle() -> None:
    first = make_semantic_obstacle(2, "chair")
    second = make_semantic_obstacle(8, "person")
    grid = make_grid(
        size=21,
        points=np.array([[9.5, 10.5], [11.5, 10.5]]),
        track_ids=np.array([2, 8]),
        semantics=(first, second),
    )

    result = solve_tangential_laplace_guidance_field(
        grid,
        guidance_policy=make_guidance_policy(),
        semantic_obstacles=(first, second),
        config=TangentialLaplaceGuidanceFieldConfig(
            tangential_buffer_radius=3.0,
            allow_dense_scene_arbitration=True,
            convergence_tolerance=1e-6,
            maximum_iterations=10_000,
            residual_check_interval=5,
        ),
    )

    assert result.obstacle_interface_mask[10, 10]
    np.testing.assert_allclose(
        result.obstacle_normals_robot_xy[10, 10],
        0.0,
    )
    np.testing.assert_allclose(result.vectors_robot_xy[10, 10], 0.0)


def test_buffer_reaching_outer_boundary_is_rejected() -> None:
    semantic = make_semantic_obstacle(4, "chair")
    grid = make_grid(
        points=np.array([[2.5, 8.5]]),
        track_ids=np.array([4]),
        semantics=(semantic,),
    )

    with pytest.raises(ValueError, match="outer grid boundary"):
        solve_tangential_laplace_guidance_field(
            grid,
            guidance_policy=make_guidance_policy(),
            semantic_obstacles=(semantic,),
            config=make_config(),
        )


def test_too_small_enabled_buffer_is_rejected() -> None:
    semantic = make_semantic_obstacle(4, "chair")
    grid = make_grid(
        points=np.array([[8.5, 8.5]]),
        track_ids=np.array([4]),
        semantics=(semantic,),
    )

    with pytest.raises(ValueError, match="at least two grid cells"):
        solve_tangential_laplace_guidance_field(
            grid,
            guidance_policy=make_guidance_policy(),
            semantic_obstacles=(semantic,),
            config=make_config(buffer_radius=1.5),
        )


def test_result_arrays_are_read_only() -> None:
    result = solve_single_obstacle()

    assert not result.vectors_robot_xy.flags.writeable
    assert not result.tangential_interface_mask.flags.writeable
    assert not result.tangential_normals_robot_xy.flags.writeable
    assert not result.obstacle_component_labels.flags.writeable

    with pytest.raises(ValueError):
        result.tangential_interface_mask[8, 5] = False


@pytest.mark.parametrize(
    "config_values",
    [
        {"tangential_buffer_radius": 0.0},
        {"tangential_buffer_radius": -1.0},
        {"tangential_buffer_radius": float("inf")},
        {"tangential_buffer_radius": float("nan")},
    ],
)
def test_invalid_buffer_configuration_is_rejected(
    config_values: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        TangentialLaplaceGuidanceFieldConfig(**config_values)


def test_non_numeric_buffer_configuration_is_rejected() -> None:
    with pytest.raises(TypeError, match="real number"):
        TangentialLaplaceGuidanceFieldConfig(
            tangential_buffer_radius="wide"
        )


def test_duplicate_semantic_track_ids_are_rejected() -> None:
    semantic = make_semantic_obstacle(4, "chair")
    grid = make_grid(
        points=np.array([[8.5, 8.5]]),
        track_ids=np.array([4]),
        semantics=(semantic,),
    )

    with pytest.raises(ValueError, match="duplicate track ID"):
        solve_tangential_laplace_guidance_field(
            grid,
            guidance_policy=make_guidance_policy(),
            semantic_obstacles=(semantic, semantic),
            config=make_config(),
        )


def test_nonconvergence_is_reported() -> None:
    config = replace(
        make_config(),
        maximum_iterations=1,
        convergence_tolerance=1e-14,
    )
    semantic = make_semantic_obstacle(4, "chair")
    grid = make_grid(
        points=np.array([[8.5, 8.5]]),
        track_ids=np.array([4]),
        semantics=(semantic,),
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        solve_tangential_laplace_guidance_field(
            grid,
            guidance_policy=make_guidance_policy(),
            semantic_obstacles=(semantic,),
            config=config,
        )
