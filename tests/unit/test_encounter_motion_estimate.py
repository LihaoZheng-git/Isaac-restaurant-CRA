from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot.safety.encounter_motion_estimate import (
    EncounterMotionEstimate, EncounterMotionEstimateConfig, EncounterPolicyMotion,
)
from robot.safety.encounter_motion import EncounterMotionClassifier
from robot.safety.obstacle_tracking import TrackedObstacle


def observation(t, xy, velocity=(.5, 0.), **changes):
    track = TrackedObstacle(7, tuple(xy), tuple(xy), velocity, velocity,
                           30, 10, True)
    return SimpleNamespace(timestamp_seconds=t,
        tracked_obstacles=(replace(track, **changes),))


@pytest.mark.parametrize('changes', [dict(window_seconds=2.), dict(minimum_samples=2),
    dict(minimum_points=True), dict(maximum_fit_residual_m=np.nan),
    dict(minimum_span_seconds=.4), dict(maximum_gap_seconds=0.)])
def test_invalid_motion_fit_configuration(changes):
    with pytest.raises(ValueError):
        EncounterMotionEstimateConfig(**changes)


@pytest.mark.parametrize('yaw', [0., 1.1])
@pytest.mark.parametrize('sign', [-1, 1])
def test_fit_uses_original_source_epochs_and_rotates_with_current_positions(yaw, sign):
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    fit = EncounterMotionEstimate()
    expected = rotation @ [sign*.5, 0.]
    for t in np.arange(5)*.1+10.:
        values, audit = fit.update(observation(t, np.array([20., 30.])+expected*(t-10.), tuple(expected)), (7,))
    np.testing.assert_allclose(values[7], expected, atol=1e-12)
    row = audit['tracks'][7]
    assert row['original_source_timestamps_seconds'] == pytest.approx([10.,10.1,10.2,10.3,10.4])
    assert row['source_timestamp_seconds'] == 10.4
    assert row['source_span_seconds'] == pytest.approx(.4)
    assert row['status'] == 'inferred'


@pytest.mark.parametrize('failure', ['identity', 'sparse', 'gap', 'missing'])
def test_loss_or_gap_never_borrows_an_old_direction(failure):
    fit = EncounterMotionEstimate()
    for i in range(5):
        fit.update(observation(1.+i*.1, (i*.05,0.)), (7,))
    if failure == 'gap':
        values, audit = fit.update(observation(1.7, (.35,0.)), (7,))
    elif failure == 'missing':
        fit.update(SimpleNamespace(timestamp_seconds=1.5, tracked_obstacles=()), ())
        values, audit = fit.update(observation(1.6, (.3,0.)), (7,))
    else:
        args = dict(matched_previous_track=False) if failure == 'identity' else dict(point_count=19)
        values, audit = fit.update(observation(1.5, (.25,0.), **args), (7,))
    assert not values and not audit['tracks'][7]['accepted']
    if failure in {'identity','sparse'}:
        assert 7 not in fit.history


@pytest.mark.parametrize('failure', ['outlier','reversal','speed'])
def test_inconsistent_or_reversed_current_observation_cannot_use_the_fit(failure):
    fit = EncounterMotionEstimate()
    for i in range(4):
        fit.update(observation(1.+i*.1, (i*.05,0.)), (7,))
    point = (.2,.3) if failure == 'outlier' else (.2,0.)
    velocity = (-.5,0.) if failure == 'reversal' else (3.,0.) if failure == 'speed' else (.5,0.)
    source = observation(1.4,point,velocity)
    before = source.tracked_obstacles[0]
    values, audit = fit.update(source, (7,))
    assert not values and not audit['tracks'][7]['accepted']
    assert source.tracked_obstacles[0] is before
    expected = {'outlier':'inconsistent_position_fit', 'reversal':'current_velocity_opposes_window_fit',
                'speed':'speed_outside_validated_scope'}
    assert audit['tracks'][7]['reason'] == expected[failure]


def test_short_centroid_shift_no_longer_reverses_encounter_policy_but_raw_motion_is_preserved():
    fit = EncounterMotionEstimate()
    raw_classifier, fitted_classifier = EncounterMotionClassifier(), EncounterMotionClassifier()
    for i in range(14):
        t = 1.+i*.1
        position = np.array([.5*(t-1.), 0.])
        raw = (.5, 0.)
        if i == 10:
            position += [-.04,-.06]
            raw = (.1,-.34)
        source = observation(t, position, raw)
        values, audit = fit.update(source, (7,))
        policy = SimpleNamespace(timestamp_seconds=t, tracked_obstacles=tuple(
            replace(track, velocity_world_xy=values[7]) if 7 in values else track
            for track in source.tracked_obstacles))
        kwargs = dict(robot_xy=(0.,0.), goal_xy=(0.,4.), human_track_ids=(7,),
                      nominal_route_direction_world_xy=(0.,1.))
        original = raw_classifier.update(source, **kwargs)['tracks'][7]
        current = fitted_classifier.update(policy, **kwargs)['tracks'][7]
        if i == 10:
            assert original['classification'] == 'uncertain'
            assert current['classification'] == 'crossing'
            assert audit['tracks'][7]['accepted']
            assert source.tracked_obstacles[0].velocity_world_xy == raw


def test_source_epoch_repeats_and_nonfinite_positions_are_rejected():
    fit = EncounterMotionEstimate()
    fit.update(observation(1.,(0.,0.)),(7,))
    with pytest.raises(ValueError, match='strictly increase'):
        fit.update(observation(1.,(0.,0.)),(7,))
    with pytest.raises(ValueError, match='finite current'):
        fit.update(observation(1.1,(np.nan,0.)),(7,))


def policy_record():
    fit = EncounterMotionEstimate()
    for i in range(5):
        source = observation(1.+i*.1, (i*.05, 0.))
        _, audit = fit.update(source, (7,))
    return EncounterPolicyMotion.from_diagnostic(audit['tracks'][7], fit.config), source


@pytest.mark.parametrize('failure', ['epoch','identity','position','raw_velocity','points','unmatched'])
def test_policy_motion_cannot_be_used_for_another_track_epoch_or_geometry(failure):
    estimate, source = policy_record()
    track = source.tracked_obstacles[0]
    estimate.validate_current_track(track, source.timestamp_seconds)
    if failure == 'epoch':
        with pytest.raises(ValueError, match='original current'):
            estimate.validate_current_track(track, 1.5)
        return
    changes = dict(identity=dict(track_id=8), position=dict(centroid_world_xy=(.3,0.)),
        raw_velocity=dict(velocity_world_xy=(.4,0.)), points=dict(point_count=29),
        unmatched=dict(matched_previous_track=False))
    with pytest.raises(ValueError, match='original current'):
        estimate.validate_current_track(replace(track, **changes[failure]), source.timestamp_seconds)


@pytest.mark.parametrize('changes', [dict(velocity_world_xy=(.6,0.)),
    dict(source_timestamps_seconds=(1.,1.1,1.2,1.3,1.5)),
    dict(raw_velocity_world_xy=(-.5,0.)),dict(point_count=19)])
def test_fabricated_or_unqualified_policy_motion_rejected(changes):
    estimate, _ = policy_record()
    with pytest.raises(ValueError):
        replace(estimate, **changes)
