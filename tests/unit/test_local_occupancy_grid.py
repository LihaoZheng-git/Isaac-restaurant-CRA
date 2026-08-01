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
from robot.safety.semantic_obstacle import SemanticObstacle


def make_policy() -> ClassSafetyPolicy:
    """Return a deterministic policy for occupancy-grid tests."""

    return ClassSafetyPolicy(
        {
            "unknown": ClassSafetyParameters(
                risk=1.0,
                boundary_flux_magnitude=6.0,
            ),
            "chair": ClassSafetyParameters(
                risk=0.4,
                boundary_flux_magnitude=3.0,
            ),
            "person": ClassSafetyParameters(
                risk=0.9,
                boundary_flux_magnitude=5.0,
            ),
        }
    )


def make_config(
    *,
    inflation_radius: float = 0.0,
) -> LocalOccupancyGridConfig:
    """Return a small grid with easy-to-check cell coordinates."""

    return LocalOccupancyGridConfig(
        resolution=0.5,
        minimum_x=-1.0,
        maximum_x=2.0,
        minimum_y=-1.5,
        maximum_y=1.5,
        obstacle_inflation_radius=inflation_radius,
    )


def make_semantic_obstacle(
    track_id: int,
    semantic_class: str,
) -> SemanticObstacle:
    """Return one semantic wrapper for a deterministic track."""

    tracked_obstacle = TrackedObstacle(
        track_id=track_id,
        centroid_world_xy=(10.0, 5.0),
        centroid_robot_xy=(1.0, 0.0),
        velocity_world_xy=(0.0, 0.0),
        velocity_robot_xy=(0.0, 0.0),
        point_count=1,
        observation_count=1,
        matched_previous_track=False,
    )
    return SemanticObstacle(
        tracked_obstacle=tracked_obstacle,
        semantic_class=semantic_class,
        semantic_confidence=1.0,
    )


def test_config_computes_expected_grid_shape() -> None:
    config = make_config()

    assert config.row_count == 6
    assert config.column_count == 6
    assert config.shape == (6, 6)


def test_rasterizes_point_into_expected_cell() -> None:
    grid = build_local_occupancy_grid(
        np.array([[0.2, -0.2]], dtype=np.float32),
        class_safety_policy=make_policy(),
        config=make_config(),
    )

    assert grid.robot_point_to_cell(np.array([0.2, -0.2])) == (2, 2)
    assert grid.occupied_mask[2, 2]
    assert grid.occupied_cell_count == 1
    assert grid.input_point_count == 1
    assert grid.rasterized_point_count == 1
    assert grid.discarded_point_count == 0


def test_cell_center_inverts_cell_geometry() -> None:
    grid = build_local_occupancy_grid(
        np.empty((0, 2)),
        class_safety_policy=make_policy(),
        config=make_config(),
    )

    assert grid.cell_center_robot_xy(2, 2) == pytest.approx(
        (0.25, -0.25)
    )
    assert grid.robot_point_to_cell(np.array([0.25, -0.25])) == (2, 2)


def test_half_open_bounds_are_applied() -> None:
    grid = build_local_occupancy_grid(
        np.array(
            [
                [-1.0, -1.5],
                [1.999, 1.499],
                [2.0, 0.0],
                [0.0, 1.5],
            ]
        ),
        class_safety_policy=make_policy(),
        config=make_config(),
    )

    assert grid.rasterized_point_count == 2
    assert grid.discarded_point_count == 2
    assert grid.robot_point_to_cell(np.array([-1.0, -1.5])) == (0, 0)
    assert grid.robot_point_to_cell(np.array([2.0, 0.0])) is None


def test_semantic_track_parameters_are_rasterized() -> None:
    grid = build_local_occupancy_grid(
        np.array([[0.2, -0.2], [1.2, 0.8]]),
        point_track_ids=np.array([3, 7]),
        semantic_obstacles=(
            make_semantic_obstacle(3, "chair"),
            make_semantic_obstacle(7, "person"),
        ),
        class_safety_policy=make_policy(),
        config=make_config(),
    )

    chair_cell = (2, 2)
    person_cell = (4, 4)

    assert grid.cell_track_ids[chair_cell] == 3
    assert grid.cell_risks[chair_cell] == pytest.approx(0.4)
    assert grid.cell_boundary_fluxes[chair_cell] == pytest.approx(-3.0)
    assert grid.cell_track_ids[person_cell] == 7
    assert grid.cell_risks[person_cell] == pytest.approx(0.9)
    assert grid.cell_boundary_fluxes[person_cell] == pytest.approx(-5.0)


@pytest.mark.parametrize("track_id", [-1, 8])
def test_missing_semantics_use_unknown_policy(track_id: int) -> None:
    grid = build_local_occupancy_grid(
        np.array([[0.2, -0.2]]),
        point_track_ids=np.array([track_id]),
        class_safety_policy=make_policy(),
        config=make_config(),
    )

    cell = (2, 2)
    assert grid.cell_track_ids[cell] == track_id
    assert grid.cell_risks[cell] == pytest.approx(1.0)
    assert grid.cell_boundary_fluxes[cell] == pytest.approx(-6.0)


def test_obstacle_inflation_uses_circular_cell_offsets() -> None:
    grid = build_local_occupancy_grid(
        np.array([[0.2, -0.2]]),
        class_safety_policy=make_policy(),
        config=make_config(inflation_radius=0.5),
    )

    expected_cells = {
        (1, 2),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 2),
    }
    occupied_cells = {
        tuple(index)
        for index in np.argwhere(grid.occupied_mask)
    }

    assert occupied_cells == expected_cells
    assert np.all(grid.cell_risks[grid.occupied_mask] == 1.0)
    assert np.all(grid.cell_boundary_fluxes[grid.occupied_mask] == -6.0)


def test_overlap_retains_more_conservative_policy() -> None:
    grid = build_local_occupancy_grid(
        np.array([[0.1, 0.1], [0.2, 0.2]]),
        point_track_ids=np.array([2, 9]),
        semantic_obstacles=(
            make_semantic_obstacle(2, "chair"),
            make_semantic_obstacle(9, "person"),
        ),
        class_safety_policy=make_policy(),
        config=make_config(),
    )

    cell = (3, 2)
    assert grid.occupied_cell_count == 1
    assert grid.cell_track_ids[cell] == 9
    assert grid.cell_risks[cell] == pytest.approx(0.9)
    assert grid.cell_boundary_fluxes[cell] == pytest.approx(-5.0)


def test_tracked_class_wins_exact_tie_against_untracked_point() -> None:
    grid = build_local_occupancy_grid(
        np.array([[0.1, 0.1], [0.2, 0.2]]),
        point_track_ids=np.array([-1, 4]),
        semantic_obstacles=(
            make_semantic_obstacle(4, "unknown"),
        ),
        class_safety_policy=make_policy(),
        config=make_config(),
    )

    assert grid.cell_track_ids[3, 2] == 4


def test_empty_input_returns_shaped_read_only_arrays() -> None:
    grid = build_local_occupancy_grid(
        np.empty((0, 2), dtype=np.float32),
        class_safety_policy=make_policy(),
        config=make_config(),
    )

    assert grid.shape == (6, 6)
    assert not np.any(grid.occupied_mask)
    assert np.all(grid.cell_track_ids == -1)
    assert np.all(grid.cell_risks == 0.0)
    assert np.all(grid.cell_boundary_fluxes == 0.0)
    assert not grid.occupied_mask.flags.writeable
    assert not grid.cell_track_ids.flags.writeable
    assert not grid.cell_risks.flags.writeable
    assert not grid.cell_boundary_fluxes.flags.writeable


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((3, 3)),
        np.array([[np.nan, 0.0]]),
    ],
)
def test_invalid_points_are_rejected(points: np.ndarray) -> None:
    with pytest.raises(ValueError):
        build_local_occupancy_grid(
            points,
            class_safety_policy=make_policy(),
            config=make_config(),
        )


def test_misaligned_track_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="point_track_ids"):
        build_local_occupancy_grid(
            np.zeros((2, 2)),
            point_track_ids=np.array([1]),
            class_safety_policy=make_policy(),
            config=make_config(),
        )


def test_noninteger_track_ids_are_rejected() -> None:
    with pytest.raises(TypeError, match="integers"):
        build_local_occupancy_grid(
            np.zeros((1, 2)),
            point_track_ids=np.array([1.0]),
            class_safety_policy=make_policy(),
            config=make_config(),
        )


def test_duplicate_semantic_track_ids_are_rejected() -> None:
    duplicate_track = make_semantic_obstacle(3, "chair")

    with pytest.raises(ValueError, match="duplicate track ID"):
        build_local_occupancy_grid(
            np.zeros((1, 2)),
            point_track_ids=np.array([3]),
            semantic_obstacles=(duplicate_track, duplicate_track),
            class_safety_policy=make_policy(),
            config=make_config(),
        )


@pytest.mark.parametrize(
    "config_values",
    [
        {"resolution": 0.0},
        {"minimum_x": 1.0, "maximum_x": 1.0},
        {"minimum_y": 1.0, "maximum_y": 0.0},
        {"obstacle_inflation_radius": -0.1},
        {"resolution": 0.7},
    ],
)
def test_invalid_grid_configuration_is_rejected(
    config_values: dict[str, float],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        LocalOccupancyGridConfig(**config_values)


def test_invalid_cell_index_is_rejected() -> None:
    grid = build_local_occupancy_grid(
        np.empty((0, 2)),
        class_safety_policy=make_policy(),
        config=make_config(),
    )

    with pytest.raises(IndexError):
        grid.cell_center_robot_xy(6, 0)
    with pytest.raises(TypeError):
        grid.cell_center_robot_xy(True, 0)
