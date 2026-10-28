import math
import json

import numpy as np
import pytest

from robot.control.planar_velocity_command import bounded_body_velocity, fixed_to_body_velocity
from robot.evaluation.quadruped_feasibility import root_observation, summarize_standing, summarize_tracking


def test_lateral_command_survives_frame_conversion():
    assert np.allclose(fixed_to_body_velocity([.2, .1, .3], math.pi/2), [.1, -.2, .3])
    assert np.allclose(fixed_to_body_velocity([0, .15, 0], 0), [0, .15, 0])


def test_command_limits_reject_rather_than_clip():
    assert np.allclose(bounded_body_velocity([.15, .1, .3]), [.15, .1, .3])
    for command in [[.25, .25, 0], [0, 0, .81], [float('nan'), 0, 0], [0, 0]]:
        with pytest.raises(ValueError):
            bounded_body_velocity(command)


def rows(phase='standing', duration=3.2, command=(0, 0, 0)):
    return [dict(time=i*.02, phase_elapsed=i*.02, phase=phase, command_body=list(command),
        position_world=[0., 0., .5], roll_pitch_yaw=[0., 0., 0.], velocity_body=list(command))
        for i in range(round(duration/.02)+1)]


def test_standing_requires_observed_stationarity_and_zero_commands():
    data = rows()
    assert summarize_standing(data)['passed']
    data[-1]['position_world'][0] = .021
    assert not summarize_standing(data)['passed']
    data[-1]['position_world'][0] = 0
    data[10]['command_body'][1] = .1
    assert not summarize_standing(data)['passed']


def test_standing_rejects_short_or_duplicate_observations():
    assert not summarize_standing(rows(duration=1.))['passed']
    data = rows()
    data[10]['time'] = data[9]['time']
    with pytest.raises(ValueError):
        summarize_standing(data)


def test_stopping_includes_braking_displacement():
    data = []
    for axis, command in [('forward', (.2, 0, 0)), ('lateral', (0, .15, 0)),
                          ('yaw', (0, 0, .3)), ('combined', (.15, .1, 0))]:
        data += rows('move_'+axis, command=command)+rows('stop_'+axis)
    assert summarize_tracking(data)['passed']
    data[-1]['position_world'][0] = .021
    report = summarize_tracking(data)
    assert not report['passed']
    assert not report['phases'][-1]['checks']['original_two_cm_stop_gate']


def test_tracking_cannot_pass_with_only_yaw_or_forward_channels_correct():
    data = rows('move_lateral', command=(0, .15, 0))
    for row in data:
        row['velocity_body'] = [0, 0, 0]
    assert not summarize_tracking(data)['passed']


def test_tracking_rejects_duplicate_epochs_and_missing_command_suite():
    data = rows('move_lateral', command=(0, .15, 0))
    assert not summarize_tracking(data)['complete_command_suite']
    data[10]['phase_elapsed'] = data[9]['phase_elapsed']
    with pytest.raises(ValueError):
        summarize_tracking(data)


def test_native_float32_observations_are_strict_json_and_owned():
    data = rows()[0]
    for key in ('command_body', 'position_world', 'roll_pitch_yaw', 'velocity_body'):
        data[key] = np.asarray(data[key], dtype=np.float32)
    converted = root_observation(**data)
    assert json.loads(json.dumps(converted, allow_nan=False)) == converted
    data['velocity_body'][0] = .2
    assert converted['velocity_body'][0] == 0


def test_native_invalid_observation_cannot_be_laundered_into_report():
    data = rows()[0]
    data['velocity_body'][0] = float('nan')
    with pytest.raises(ValueError):
        root_observation(**data)
