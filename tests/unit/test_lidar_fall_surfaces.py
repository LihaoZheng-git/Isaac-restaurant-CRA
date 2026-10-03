import numpy as np
import pytest

from robot.perception.fall_height import estimate_fall_height
from robot.perception.lidar_fall_surfaces import (
    LidarFallSurfaceConfig,
    estimate_lidar_fall_surfaces,
)


CONFIG = LidarFallSurfaceConfig(
    minimum_surface_points=4,
    direction_count=16,
    surface_band_tolerance_m=0.025,
    height_uncertainty_m=0.02,
)


def plane(z, x_range, y_range, count=31):
    xs = np.linspace(*x_range, count)
    ys = np.linspace(*y_range, count)
    return np.asarray([(x, y, z) for x in xs for y in ys], dtype=float)


def object_points(center=(0, 0), bottom=0.84):
    x, y = center
    return np.asarray(
        [
            (x + dx, y + dy, bottom + dz)
            for dx in (-0.08, 0, 0.08)
            for dy in (-0.08, 0, 0.08)
            for dz in (0, 0.08, 0.16)
        ],
        dtype=float,
    )


def test_table_edge_and_floor_produce_directional_surface_drop():
    table = plane(0.8, (-0.8, 0.18), (-0.7, 0.7))
    floor = plane(0.0, (0.3, 1.2), (-0.4, 0.4))
    item = object_points()
    result = estimate_lidar_fall_surfaces(
        np.vstack((table, floor, item)),
        item,
        timestamp_seconds=2,
        config=CONFIG,
    )
    assert result.is_available
    assert result.fall_direction_robot_xy[0] > 0.5
    fall = estimate_fall_height(result.evidence, timestamp_seconds=2)
    assert fall.value == pytest.approx(0.8, abs=0.03)
    assert "not object height" in fall.evidence[-1]


def test_floor_supported_object_has_zero_local_fall_height():
    floor = plane(0.0, (-1, 1), (-1, 1))
    item = object_points(bottom=0.04)
    result = estimate_lidar_fall_surfaces(
        np.vstack((floor, item)), item, timestamp_seconds=2, config=CONFIG
    )
    assert result.is_available
    assert result.fall_direction_robot_xy is None
    assert estimate_fall_height(result.evidence, timestamp_seconds=2).value == 0


def test_table_center_without_visible_lower_landing_is_unavailable():
    table = plane(0.8, (-1.5, 1.5), (-1.5, 1.5))
    item = object_points()
    result = estimate_lidar_fall_surfaces(
        np.vstack((table, item)), item, timestamp_seconds=2, config=CONFIG
    )
    assert result.is_available
    assert estimate_fall_height(result.evidence, timestamp_seconds=2).value == 0


def test_missing_support_stays_unavailable():
    item = object_points()
    result = estimate_lidar_fall_surfaces(
        item, item, timestamp_seconds=2, config=CONFIG
    )
    assert not result.is_available
    assert "support" in result.rejection_reasons[0]


def test_partial_support_coverage_without_landing_does_not_imply_zero_drop():
    table = plane(0.8, (-0.8, 0.18), (-0.7, 0.7))
    item = object_points()
    result = estimate_lidar_fall_surfaces(
        np.vstack((table, item)), item, timestamp_seconds=2, config=CONFIG,
    )
    assert not result.is_available
    assert "landing" in result.rejection_reasons[0]


def test_small_surrounding_support_patch_without_farther_visibility_is_unknown():
    # The patch spans every direction, but its nearby edges have no visible
    # landing. Direction count alone cannot justify a zero drop.
    support = plane(0.8, (-0.45, 0.45), (-0.45, 0.45), count=51)
    item = object_points()
    result = estimate_lidar_fall_surfaces(np.vstack((support, item)), item,
                                         timestamp_seconds=2, config=CONFIG)
    assert not result.is_available
    assert "landing" in result.rejection_reasons[0]


def test_dense_object_remnants_do_not_hide_a_visible_support_plane():
    item = object_points()
    table = plane(0.8, (-0.8, 0.18), (-0.7, 0.7), count=15)
    # Untracked returns from a narrow object face outnumber the support band.
    remnants = plane(0.85, (-0.04, 0.04), (-0.04, 0.04), count=41)
    floor = plane(0.0, (0.3, 1.2), (-0.4, 0.4))
    result = estimate_lidar_fall_surfaces(
        np.vstack((table, remnants, floor, item)), item,
        timestamp_seconds=2, config=CONFIG,
    )
    assert result.is_available
    assert result.evidence.support_height_m.value == pytest.approx(0.8)


def test_rotated_vertical_surface_band_does_not_establish_planar_support():
    item = object_points()
    values = np.linspace(-0.5, 0.5, 100)
    line = np.column_stack((values, values, np.full_like(values, 0.8)))
    result = estimate_lidar_fall_surfaces(
        np.vstack((line, item)), item, timestamp_seconds=2, config=CONFIG,
    )
    assert not result.is_available
    assert "planar support" in result.rejection_reasons[0]


def test_ambiguous_opposing_drops_stay_unavailable():
    table = plane(0.8, (-0.18, 0.18), (-0.8, 0.8))
    left_floor = plane(0.0, (-1.2, -0.3), (-0.3, 0.3))
    right_floor = plane(0.0, (0.3, 1.2), (-0.3, 0.3))
    item = object_points()
    result = estimate_lidar_fall_surfaces(
        np.vstack((table, left_floor, right_floor, item)),
        item,
        timestamp_seconds=2,
        config=CONFIG,
    )
    assert not result.is_available
    assert "ambiguous" in result.rejection_reasons[0]
