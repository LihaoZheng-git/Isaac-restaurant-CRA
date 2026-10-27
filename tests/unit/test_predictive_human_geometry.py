from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE
from robot.safety.predictive_human_geometry import (
    HumanEnvelopePrediction, HumanGeometryForecastConfig, PredictiveHumanGeometry,
    PredictiveHumanGeometryMemory,
)
from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint


FOOTPRINT = RectangularRobotFootprint(.04, .687351, .2466, .2466, .05)


def prediction(center=(1.5, 0.), velocity=(-.5, 0.), *, key=7, config=None):
    config = config or HumanGeometryForecastConfig()
    times = np.linspace(.5, 1., 6)
    points = np.array(center)+(times-1.)[:, None]*velocity
    return HumanEnvelopePrediction(key, tuple(times), tuple(map(tuple, points)), velocity,
        .65, .8, 42, 100, config)


def geometry(forecast=None, *, anchor=(0., 0.), yaw=0.):
    f = forecast or prediction()
    return PredictiveHumanGeometry(60, 1., anchor, yaw, (f.track_id,), (f,), f.config)


def test_future_motion_is_an_additional_margin_without_erasing_the_current_body():
    states, times = np.zeros((2, 3)), np.array([0., 1.2])
    moving, _ = geometry().constraints(states, times, FOOTPRINT, .1)
    static, _ = geometry(prediction(velocity=(0., 0.))).constraints(states, times, FOOTPRINT, .1)
    assert moving[0, 0] == pytest.approx(static[0, 0])
    assert moving[0, 0] > 0 and moving[1, 0] < 0 < static[1, 0]
    assert geometry().diagnostic()['status'] == 'prior'
    assert prediction().diagnostic()['velocity_basis'].startswith('inferred_')


@pytest.mark.parametrize('state', [[.1, -.2, .3], [.3, .2, -.8], [1.1, .1, .2], [2., .4, 1.2]])
def test_oriented_rectangle_state_jacobian_matches_independent_central_differences(state):
    g = geometry(prediction(center=(2., .7), velocity=(-.3, .1)))
    state = np.array(state)
    _, gradient = g.constraints(state[None], np.array([.4]), FOOTPRINT, .1)
    eps = 1e-6
    actual = []
    for axis in range(3):
        delta = np.eye(3)[axis]*eps
        high = g.constraints((state+delta)[None], np.array([.4]), FOOTPRINT, .1)[0]
        low = g.constraints((state-delta)[None], np.array([.4]), FOOTPRINT, .1)[0]
        actual.append(float((high-low).item())/(2*eps))
    np.testing.assert_allclose(gradient[0, 0], actual, atol=2e-7, rtol=1e-6)


def test_geometry_is_invariant_under_rigid_world_frame_changes():
    old = prediction(center=(2., .7), velocity=(-.3, .1))
    angle = .8
    r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    origin = np.array([28., 7.])
    moved = replace(old, centroids_world_xy=tuple(map(tuple, np.array(old.centroids_world_xy)@r.T+origin)),
        raw_velocity_world_xy=tuple(np.array(old.raw_velocity_world_xy)@r.T))
    states, times = np.array([[.2, .1, .3], [.4, .3, -.4]]), np.array([.2, .8])
    a = geometry(old).constraints(states, times, FOOTPRINT, .1)
    b = geometry(moved, anchor=tuple(origin), yaw=angle).constraints(states, times, FOOTPRINT, .1)
    for left, right in zip(a, b):
        np.testing.assert_allclose(left, right, atol=1e-12, rtol=0)


@pytest.mark.parametrize('times', [[-.01], [1.71], [np.nan]])
def test_no_extrapolation_beyond_the_declared_prediction_horizon(times):
    with pytest.raises(ValueError):
        geometry().constraints(np.zeros((1, 3)), np.array(times), FOOTPRINT, .1)


def test_original_source_anchor_and_identity_cannot_be_transferred_or_renewed():
    g = geometry()
    g.validate_source(60, 1., (0., 0.), 0.)
    for args in [(61, 1., (0., 0.), 0.), (60, 1.1, (0., 0.), 0.), (60, 1., (.1, 0.), 0.), (60, 1., (0., 0.), .1)]:
        with pytest.raises(ValueError):
            g.validate_source(*args)
    with pytest.raises(ValueError):
        replace(g, forecasts=(prediction(key=8),))
    with pytest.raises(ValueError):
        replace(g, forecasts=(prediction(), prediction()))


def observation(t, *, key=7, matched=True, count=100, raw=(-.5, 0.)):
    center = (2.-.5*t, 1.)
    track = NS(track_id=key, centroid_world_xy=center, velocity_world_xy=raw,
        matched_previous_track=matched, observation_count=20, point_count=count)
    tracking = NS(timestamp_seconds=t, tracked_obstacles=(track,))
    envelope = dict(track_id=key, source=HUMAN_ENVELOPE_SOURCE, status='prior',
        lidar_timestamp_seconds=t, applied_to_occupancy=True, center_world_xy=center,
        radius_m=.65, current_support_count=count, original_person_camera_timestamp=.05,
        original_person_camera_frame_id=3)
    return tracking, [envelope]


def update(memory, t, **kwargs):
    tracking, envelopes = observation(t, **kwargs)
    return memory.update(tracking, envelopes, sensor_frame_id=round(t*60), anchor=(0., 0.), anchor_yaw=0.)


def test_current_window_fit_warms_up_then_preserves_source_and_original_image_epochs():
    m = PredictiveHumanGeometryMemory()
    for t in np.arange(.1, .7, .1):
        result, audit = update(m, float(t))
    assert result.ready and len(result.forecasts) == 1
    f = result.forecasts[0]
    np.testing.assert_allclose(f.velocity_world_xy, [-.5, 0.], atol=1e-14)
    assert f.original_person_camera_timestamp == .05
    assert f.source_timestamps_seconds[0] == .1
    assert f.source_timestamp_seconds == pytest.approx(.6)
    assert audit['fits'][0]['status'] == 'prior'
    assert not update(m, 1.)[0].ready  # original gap clears, never renews old fit


@pytest.mark.parametrize('change', [dict(matched=False), dict(count=19), dict(key=8), dict(raw=(.5, 0.))])
def test_missing_support_identity_gap_or_opposing_motion_never_becomes_zero_prediction(change):
    m = PredictiveHumanGeometryMemory()
    for t in np.arange(.1, .7, .1):
        update(m, float(t))
    result, audit = update(m, .7, **change)
    assert not result.ready and not result.forecasts
    assert audit['fits'][0]['status'] == 'unavailable'
    with pytest.raises(ValueError):
        result.constraints(np.zeros((1, 3)), np.array([0.]), FOOTPRINT, .1)


def test_invalid_current_envelope_is_transactional_and_duplicate_time_rejected():
    m = PredictiveHumanGeometryMemory()
    update(m, .1)
    before = deepcopy(vars(m))
    tracking, envelopes = observation(.2)
    envelopes[0]['lidar_timestamp_seconds'] = .1
    with pytest.raises(ValueError):
        m.update(tracking, envelopes, sensor_frame_id=12, anchor=(0., 0.), anchor_yaw=0.)
    assert vars(m) == before
    with pytest.raises(ValueError):
        update(m, .1)


@pytest.mark.parametrize('changes', [dict(horizon_seconds=1.21), dict(velocity_uncertainty_prior_mps=0.),
    dict(maximum_gap_seconds=.3), dict(minimum_samples=True), dict(maximum_fit_residual_m=np.nan)])
def test_unbounded_invalid_configuration_rejected(changes):
    with pytest.raises(ValueError):
        HumanGeometryForecastConfig(**changes)
