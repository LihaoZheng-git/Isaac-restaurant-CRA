import copy

import pytest

from robot.evaluation.post_goal_observation import summarize_post_goal_observation


def samples(states, clearances):
    return dict(initial_state=states[0], final_state=states[-1], translation_drift_m=.001,
        records=[dict(state=s, clearance_m=c) for s, c in zip(states, clearances)])


def windows():
    return (samples([[.02, 0., 0.], [.021, 0., 0.]], [.4, .39]),
            samples([[.021, 0., 0.], [.022, 0., 0.], [.023, 0., 0.]], [.39, .38, .37]))


def test_post_goal_observation_keeps_original_braking_and_checks_total_trace():
    braking, tail = windows()
    before = copy.deepcopy(braking)
    result = summarize_post_goal_observation(braking, tail, goal_xy=(0., 0.), goal_tolerance=.1)
    assert result['passed'] and braking == before
    assert result['translation_drift_m'] == pytest.approx(.003)
    assert result['original_braking_drift_m'] == .001
    assert result['minimum_clearance_m'] == .37


@pytest.mark.parametrize('failure', ['temporary_collision', 'drift_then_return', 'goal_lost'])
def test_zero_final_drift_does_not_hide_intermediate_hazards_or_goal_loss(failure):
    braking, tail = windows()
    if failure == 'temporary_collision':
        tail['records'][1]['clearance_m'] = .01
    elif failure == 'drift_then_return':
        tail['records'][1]['state'] = [.08, 0., 0.]
    else:
        tail['records'][1]['state'] = [.11, 0., 0.]
    assert not summarize_post_goal_observation(braking, tail, goal_xy=(0., 0.), goal_tolerance=.1)['passed']


def test_tail_cannot_restart_at_a_different_stopping_pose():
    braking, tail = windows()
    tail['initial_state'] = [0., 0., 0.]
    with pytest.raises(ValueError, match='boundary'):
        summarize_post_goal_observation(braking, tail, goal_xy=(0., 0.), goal_tolerance=.1)
