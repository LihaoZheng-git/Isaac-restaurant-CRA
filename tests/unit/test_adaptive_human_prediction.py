from dataclasses import replace

import numpy as np
import pytest
from test_predictive_human_geometry import FOOTPRINT, geometry, prediction

from robot.safety.predictive_human_geometry import HumanGeometryForecastConfig


@pytest.mark.parametrize('raw,expected', [((-.5, 0.), .3), ((-.6, 0.), .4), ((-1., 0.), .5), ((0., 0.), .5)])
def test_current_raw_window_disagreement_sets_a_bounded_prior_without_changing_fit_or_sources(raw, expected):
    cfg = HumanGeometryForecastConfig(velocity_uncertainty_prior_mps=.5, velocity_uncertainty_floor_prior_mps=.3)
    f = prediction(config=cfg)
    changed = replace(f, raw_velocity_world_xy=raw)
    assert changed.uncertainty_growth_prior_mps == pytest.approx(expected)
    np.testing.assert_array_equal(changed.velocity_world_xy, f.velocity_world_xy)
    assert changed.source_timestamps_seconds == f.source_timestamps_seconds
    assert changed.original_person_camera_timestamp == f.original_person_camera_timestamp
    assert changed.radius_prior_m == f.radius_prior_m
    diagnostic = changed.diagnostic()
    assert diagnostic['status'] == 'prior'
    assert diagnostic['uncertainty_growth_basis'] == 'bounded_prior_from_current_raw_minus_window_fit_velocity'


def test_adaptive_rate_changes_only_future_radius_not_current_body_or_chassis_gradients():
    fixed = prediction(config=HumanGeometryForecastConfig(velocity_uncertainty_prior_mps=.5))
    adaptive = replace(fixed, config=replace(fixed.config, velocity_uncertainty_floor_prior_mps=.3))
    states, times = np.array([[.1, .1, .3], [.2, -.1, -.2]]), np.array([0., 1.2])
    a, ga = geometry(fixed).constraints(states, times, FOOTPRINT, .1)
    b, gb = geometry(adaptive).constraints(states, times, FOOTPRINT, .1)
    np.testing.assert_allclose((b-a).ravel(), .2*times, atol=1e-12)
    np.testing.assert_array_equal(ga, gb)
    assert a[0] == b[0]
    assert fixed.uncertainty_growth_prior_mps == .5
    assert HumanGeometryForecastConfig().velocity_uncertainty_floor_prior_mps is None


@pytest.mark.parametrize('floor,ceiling', [(.29, .5), (.6, .5), (.3, .25), (True, .5), (float('nan'), .5), (.3, 1.01)])
def test_adaptive_prior_requires_explicit_bounded_floor_and_ceiling(floor, ceiling):
    with pytest.raises(ValueError):
        HumanGeometryForecastConfig(velocity_uncertainty_floor_prior_mps=floor, velocity_uncertainty_prior_mps=ceiling)


def test_invalid_opposing_motion_does_not_become_a_smaller_uncertainty():
    cfg = HumanGeometryForecastConfig(velocity_uncertainty_floor_prior_mps=.3, velocity_uncertainty_prior_mps=.5)
    with pytest.raises(ValueError):
        replace(prediction(config=cfg), raw_velocity_world_xy=(.5, 0.))
