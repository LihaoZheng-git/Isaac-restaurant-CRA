from dataclasses import replace

import numpy as np
import pytest
from test_predictive_human_geometry import geometry, prediction

from robot.safety.parallel_preparation import ParallelPreparation, ParallelPreparationConfig
from robot.safety.predictive_human_geometry import PredictiveHumanGeometry


def current(t=1., *, center=(3.5, 1.), velocity=(0., 0.), key=7):
    f = prediction(center=center, velocity=velocity, key=key)
    times = np.array(f.source_timestamps_seconds)+t-1.
    times[-1] = t
    f = replace(f, source_timestamps_seconds=tuple(times))
    return PredictiveHumanGeometry(round(60*t), t, (0., 0.), 0., (key,), (f,), f.config)


def step(memory, t=1., *, fresh=True, g=None):
    cue = dict(basis='fresh_same_frame_person_SAM_association; class cue only',
        camera_frame_id=round(10*t), camera_timestamp_seconds=t+.1)
    return memory.update(g or current(t), robot_xy=(0., 0.), route_direction_world_xy=(1., 0.),
        visual_evidence={7: cue} if fresh else {}, camera_timestamp_seconds=t+.1, camera_frame_id=round(10*t))


def test_current_quiet_person_can_prepare_without_claiming_walking_intent():
    sides, audit = step(ParallelPreparation())
    assert sides[7].value == 'right'
    assert audit['qualified_candidates'][0]['basis'] == 'quiet_current_person_geometry'
    assert audit['status'] == 'prior'
    assert audit['original_selection']['selected_timestamp_seconds'] == 1.
    assert 'no walking-intent inference' in audit['scope']


def test_no_new_selection_from_old_identity_without_current_visual_cue():
    sides, audit = step(ParallelPreparation(), fresh=False)
    assert not sides and audit['decision'] == 'no_fresh_visual_person_for_new_selection'


def test_hold_keeps_original_visual_and_policy_clocks_and_expiry_requires_new_selection():
    m = ParallelPreparation(ParallelPreparationConfig(maximum_selection_seconds=.2))
    _, first = step(m)
    _, held = step(m, 1.1)
    assert first['original_selection'] == held['original_selection']
    assert step(m, 1.3)[0] == {}
    assert step(m, 1.4, fresh=False)[0] == {}
    _, renewed = step(m, 1.5)
    assert renewed['original_selection']['selected_timestamp_seconds'] == 1.5


@pytest.mark.parametrize('center,velocity', [((-1., 1.), (0., 0.)), ((3.9, 1.), (0., 0.)),
    ((2., .1), (0., 0.)), ((2., 1.), (.5, 0.)), ((2., 1.), (0., .5))])
def test_ambiguous_side_behind_outside_and_nonparallel_motion_never_get_preparation(center, velocity):
    assert step(ParallelPreparation(), g=current(center=center, velocity=velocity))[0] == {}


def test_missing_fit_multiple_people_and_source_gap_release_without_same_frame_reselection():
    m = ParallelPreparation()
    step(m)
    assert not step(m, 1.1, g=replace(current(1.1), forecasts=()))[0]
    first, second = current().forecasts[0], current(center=(3.5, -1.), key=8).forecasts[0]
    ambiguous = replace(geometry(first), required_track_ids=(7, 8), forecasts=(first, second))
    assert not step(ParallelPreparation(), g=ambiguous)[0]
    m = ParallelPreparation()
    step(m)
    assert not step(m, 1.4)[0]


def test_bad_visual_epoch_and_nonunit_nominal_direction_reject_before_selection():
    m = ParallelPreparation()
    with pytest.raises(ValueError):
        m.update(current(), robot_xy=(0., 0.), route_direction_world_xy=(2., 0.), visual_evidence={},
            camera_timestamp_seconds=1.1, camera_frame_id=11)
    with pytest.raises(ValueError):
        m.update(current(), robot_xy=(0., 0.), route_direction_world_xy=(1., 0.),
            visual_evidence={7: dict(camera_frame_id=10, camera_timestamp_seconds=1., basis='old')},
            camera_timestamp_seconds=1.1, camera_frame_id=11)
    assert m.encounter is None
