"""Independent ground/support controls; unobserved continuity stays a prior."""

from dataclasses import replace

import numpy as np
import pytest

from robot.perception.ground_support import (
    GroundSupportConfig,
    estimate_spherical_ground_support,
)
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate
from robot.perception.spherical_mass import estimate_spherical_geometry


def sphere(radius=0.20, bottom=0.0, timestamp=2.0):
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    vertical = np.linspace(-0.8, 0.8, 15)
    points = np.asarray(
        [
            (
                radius * np.sqrt(1 - z * z) * np.cos(a),
                radius * np.sqrt(1 - z * z) * np.sin(a),
                bottom + radius + radius * z,
            )
            for z in vertical
            for a in angles
        ]
    )
    return estimate_spherical_geometry(points, timestamp_seconds=timestamp), points


def floor(*, slope=0.0, z=0.0, gap=0.8):
    return np.asarray(
        [
            (x, y, z + slope * x)
            for x in np.linspace(gap, 1.7, 25)
            for y in np.linspace(-0.8, 0.8, 31)
        ]
    )


def prior(timestamp=2.0, **kwargs):
    values = dict(
        value=1,
        lower_bound=1,
        upper_bound=1,
        confidence=0.5,
        unit="1",
        source="controlled_floor_continuity",
        timestamp_seconds=timestamp,
        status=EstimateStatus.PRIOR,
    )
    return ScalarEstimate(**(values | kwargs))


def assess(*, geometry=None, scene=None, **kwargs):
    if geometry is None:
        geometry, _ = sphere()
    return estimate_spherical_ground_support(
        floor() if scene is None else scene,
        geometry,
        **(
            dict(
                timestamp_seconds=2.0,
                scene_timestamp_seconds=2.0,
                ground_continuity_prior=prior(),
                prior_scope="bounded flat-floor fixture",
            )
            | kwargs
        ),
    )


def test_observed_plane_plus_explicit_continuity_retains_prior_and_nonzero_interval():
    result = assess()
    assert not result.rejection_reasons
    assert result.floor_height_m.status is EstimateStatus.INFERRED
    assert result.fall_height_m.status is EstimateStatus.PRIOR
    assert result.fall_height_m.value == 0
    assert 0.04 < result.fall_height_m.upper_bound < 0.10
    assert result.nearest_floor_distance_m >= 0.8
    interaction = result.potential_interaction_likelihood
    assert (interaction.value, interaction.lower_bound, interaction.upper_bound) == (
        1,
        0,
        1,
    )
    assert interaction.status is EstimateStatus.PRIOR
    assert "not object height" in result.fall_height_m.evidence[-1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ground_continuity_prior": None},
        {"prior_scope": None},
        {"ground_continuity_prior": prior(confidence=0)},
        {
            "ground_continuity_prior": ScalarEstimate.unavailable(
                unit="1", source="missing", timestamp_seconds=2
            )
        },
    ],
)
def test_missing_continuity_is_unknown_even_with_good_floor_and_shape(kwargs):
    result = assess(**kwargs)
    assert result.floor_height_m.is_available
    assert not result.fall_height_m.is_available
    assert not result.potential_interaction_likelihood.is_available
    assert "prior required" in result.rejection_reasons[0]


def test_sphere_height_never_becomes_local_drop():
    results = [assess(geometry=sphere(radius=r)[0]) for r in (0.1, 0.3)]
    assert all(not result.rejection_reasons for result in results)
    assert [result.fall_height_m.value for result in results] == [0, 0]
    assert results[0].fall_height_m.upper_bound == pytest.approx(
        results[1].fall_height_m.upper_bound
    )


@pytest.mark.parametrize("bottom", [0.25, -0.2])
def test_elevated_or_buried_sphere_rejects_ground_hypothesis(bottom):
    result = assess(geometry=sphere(bottom=bottom)[0])
    assert not result.fall_height_m.is_available
    assert "inconsistent" in result.rejection_reasons[0]


def test_sphere_returns_cannot_supply_their_own_floor():
    geometry, points = sphere()
    result = assess(geometry=geometry, scene=points)
    assert not result.floor_height_m.is_available
    assert not result.fall_height_m.is_available


def test_floor_outside_continuity_scope_is_unknown():
    result = assess(config=GroundSupportConfig(maximum_extrapolation_m=0.7))
    assert not result.fall_height_m.is_available
    assert "extrapolation" in result.rejection_reasons[0]


def test_slope_outside_domain_rejects_floor():
    config = GroundSupportConfig(maximum_plane_slope=0.02)
    result = assess(scene=floor(slope=0.04), config=config)
    assert not result.fall_height_m.is_available
    assert "slope" in result.rejection_reasons[0]


def test_line_is_not_a_floor_plane():
    xs = np.linspace(0.8, 1.7, 100)
    result = assess(scene=np.column_stack((xs, xs, np.zeros_like(xs))))
    assert not result.fall_height_m.is_available
    assert "two-dimensional" in result.rejection_reasons[0]


def test_nonplanar_low_band_rejects():
    points = floor()
    points[:, 2] = np.random.default_rng(12).uniform(-0.03, 0.03, len(points))
    result = assess(
        scene=points, config=GroundSupportConfig(maximum_plane_residual_m=0.005)
    )
    assert not result.fall_height_m.is_available
    assert "planar" in result.rejection_reasons[0]


def test_very_uncertain_bottom_rejects_ground_contact():
    geometry, _ = sphere()
    geometry = replace(geometry, radius_m=replace(geometry.radius_m, upper_bound=0.35))
    result = assess(geometry=geometry)
    assert not result.fall_height_m.is_available
    assert "uncertain" in result.rejection_reasons[0]


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"timestamp_seconds": 3}, "sphere geometry"),
        ({"scene_timestamp_seconds": 1}, "scene point cloud"),
        ({"scene_timestamp_seconds": 3}, "scene point cloud"),
        ({"ground_continuity_prior": prior(timestamp=1)}, "ground-continuity prior"),
        ({"ground_continuity_prior": prior(timestamp=3)}, "ground-continuity prior"),
    ],
)
def test_future_stale_inputs_are_not_refreshed(kwargs, reason):
    result = assess(**kwargs)
    assert not result.fall_height_m.is_available
    assert reason in result.rejection_reasons[0]


def test_oldest_evidence_time_is_preserved():
    result = assess(
        ground_continuity_prior=prior(timestamp=1.8), scene_timestamp_seconds=1.9
    )
    assert result.fall_height_m.timestamp_seconds == 1.8
    assert result.floor_height_m.timestamp_seconds == 1.9


@pytest.mark.parametrize(
    "item",
    [
        prior(status=EstimateStatus.ORACLE),
        prior(status=EstimateStatus.INFERRED),
        prior(value=0.9, lower_bound=0.9),
    ],
)
def test_continuity_cannot_be_disguised_as_sensor_evidence(item):
    with pytest.raises(ValueError, match="explicit PRIOR"):
        assess(ground_continuity_prior=item)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maximum_extrapolation_m": 3},
        {"minimum_floor_points": True},
        {"floor_confidence_cap": 2},
        {"height_allowance_m": 0},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        GroundSupportConfig(**kwargs)
