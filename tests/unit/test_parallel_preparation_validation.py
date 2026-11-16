from copy import deepcopy
import json

import pytest
from test_parallel_preparation import current, step

from robot.evaluation.parallel_preparation_validation import validate_parallel_preparation
from robot.safety.parallel_preparation import ParallelPreparation, ParallelPreparationConfig


def records():
    m, result = ParallelPreparation(), []
    for t in (1., 1.1, 1.2):
        _, a = step(m, t)
        a['used_for_reference'] = True
        result.append(dict(time=t, sensor_frame_id=current(t).sensor_frame_id,
            parallel_preparation=a, motion_passing_reference_track_ids=(7,),
            motion_passing_reference_source_timestamp_seconds=1.,
            applied_field_policies=[dict(track_or_group_id=7, applied_passing_side='right', **{'lambda':1.25})]))
    return result


def test_native_and_live_layouts_accept_only_the_original_current_preparation_proof():
    rows = records()
    assert validate_parallel_preparation(rows)['passed']
    live = [dict(sensor_timestamp=r['time'], sensor_frame_id=r['sensor_frame_id'],
        diagnostics={k: v for k, v in r.items() if k not in ['time', 'sensor_frame_id']}) for r in rows]
    assert validate_parallel_preparation(live)['passed']
    assert validate_parallel_preparation(rows)['original_selection_count'] == 1
    assert json.loads(json.dumps(validate_parallel_preparation(rows), allow_nan=False))['passed']


@pytest.mark.parametrize('failure', ['wrong_frame', 'renewed_selection', 'wrong_visual', 'missing_start',
    'side', 'field_clock', 'sparse', 'unready', 'old_fit', 'wrong_preference', 'field_side', 'lambda'])
def test_missing_or_mutated_proofs_are_rejected(failure):
    rows = deepcopy(records())
    a = rows[-1]['parallel_preparation']
    if failure == 'wrong_frame':
        rows[-1]['sensor_frame_id'] += 1
    elif failure == 'renewed_selection':
        a['original_selection']['selected_timestamp_seconds'] = 1.1
    elif failure == 'wrong_visual':
        a['original_selection']['original_visual_evidence']['camera_frame_id'] += 1
    elif failure == 'missing_start':
        rows = rows[1:]
    elif failure == 'side':
        a['original_selection']['side'] = 'left'
    elif failure == 'field_clock':
        rows[-1]['motion_passing_reference_source_timestamp_seconds'] = 1.1
    elif failure == 'sparse':
        a['qualified_candidates'][0]['current_support_count'] = 19
    elif failure == 'unready':
        a['current_fit_ready'] = False
    elif failure == 'old_fit':
        a['qualified_candidates'][0]['current_fit_source_times'] = [.5, 1.1]
    elif failure == 'wrong_preference':
        a['preferences'][7] = 'left'
    elif failure == 'field_side':
        rows[-1]['applied_field_policies'][0]['applied_passing_side'] = 'left'
    elif failure == 'lambda':
        rows[-1]['applied_field_policies'][0]['lambda'] = 0.
    assert not validate_parallel_preparation(rows)['passed']


def test_exit_scope_is_audited_only_for_held_policy_not_new_selection():
    m, rows = ParallelPreparation(ParallelPreparationConfig(exit_lateral_separation_m=2.2)), []
    for i, t in enumerate((1., 1.1, 1.2)):
        cue = dict(basis='fresh same-frame person', camera_frame_id=10, camera_timestamp_seconds=1.1)
        g = current(t, center=(2., 1.59))
        _, a = m.update(g, robot_xy=(0., -.02*i), route_direction_world_xy=(1., 0.),
            visual_evidence={7: cue} if i == 0 else {}, camera_timestamp_seconds=t+.1, camera_frame_id=10+i)
        a['used_for_reference'] = True
        rows.append(dict(time=t, sensor_frame_id=g.sensor_frame_id, parallel_preparation=a,
            motion_passing_reference_track_ids=(7,), motion_passing_reference_source_timestamp_seconds=1.,
            applied_field_policies=[dict(track_or_group_id=7, applied_passing_side='right', **{'lambda': 1.25})]))
    assert validate_parallel_preparation(rows)['passed']
    wrong = deepcopy(rows)
    wrong[0]['parallel_preparation']['current_lateral_limit_m'] = 2.2
    assert not validate_parallel_preparation(wrong)['passed']
    wrong = deepcopy(rows)
    wrong[-1]['parallel_preparation']['qualification_mode'] = 'initial'
    assert not validate_parallel_preparation(wrong)['passed']
