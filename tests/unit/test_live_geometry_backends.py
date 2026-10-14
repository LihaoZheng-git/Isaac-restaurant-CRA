import numpy as np
import pytest

from robot.safety.obstacle_tracking import _cluster_point_indices
from robot.safety.tangential_laplace_guidance_field import (
    _normal_interface_data,
    _normal_interface_data_vectorized,
    _ObstacleComponent,
)
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidanceParameters
from robot.safety.class_safety_policy import ClassSafetyParameters
from test_safe_sage_cbf_filter import make_grid


@pytest.mark.parametrize("radius", [0.2, 0.5, 1.0])
def test_kdtree_clusters_match_dense_including_duplicates_chains_and_exact_boundary(
    radius,
):
    points = np.concatenate(
        (
            np.random.default_rng(15).normal(size=(300, 2)),
            np.array([[4, 4], [4, 4], [4 + radius, 4], [8, 8]]),
        )
    )
    dense = _cluster_point_indices(
        points, cluster_radius=radius, minimum_cluster_points=2
    )
    tree = _cluster_point_indices(
        points, cluster_radius=radius, minimum_cluster_points=2, backend="kdtree"
    )
    assert len(dense) == len(tree)
    for a, b in zip(dense, tree):
        np.testing.assert_array_equal(a, b)


def test_vectorized_normal_interface_matches_ordered_loop_including_dense_saddles():
    from dataclasses import replace

    grid = make_grid()
    mask = np.random.default_rng(3).random(grid.shape) > 0.7
    grid = replace(grid, occupied_mask=mask)
    labels = np.where(mask, 0, -1)
    parameter = SafeSAGEGuidanceParameters(ClassSafetyParameters(1, 4), 1.25, "right")
    components = (_ObstacleComponent(0, np.argwhere(mask), parameter),)
    legacy = _normal_interface_data(
        grid,
        component_labels=labels,
        components=components,
        allow_dense_scene_arbitration=True,
    )
    fast = _normal_interface_data_vectorized(
        grid,
        component_labels=labels,
        components=components,
        allow_dense_scene_arbitration=True,
    )
    for a, b in zip(legacy, fast):
        np.testing.assert_array_equal(a, b)


def test_spatial_index_buffer_normals_preserve_all_nearest_cell_ties():
    from robot.safety.tangential_laplace_guidance_field import (
        _distance_and_normal_sums,
        _distance_and_normal_sums_tree,
    )

    rows, columns = np.indices((20, 30))
    for occupied in (
        np.array([[10, 15]]),
        np.array([[8, 10], [8, 14], [12, 10], [12, 14]]),
        np.argwhere(np.random.default_rng(14).random((20, 30)) > 0.75),
    ):
        expected = _distance_and_normal_sums(rows, columns, occupied, resolution=0.1)
        fast = _distance_and_normal_sums_tree(rows, columns, occupied, resolution=0.1)
        for a, b in zip(expected, fast):
            np.testing.assert_array_equal(a, b)
