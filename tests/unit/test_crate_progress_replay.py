"""Offline diagnostics must not turn mismatched evidence into native validation."""
from dataclasses import asdict, replace
import hashlib
import json
import pickle

import numpy as np
import pytest
from test_selected_tangent_reference import planner, runtime_field

from robot.evaluation.crate_progress_diagnosis import (
    load, nominal_for, probe, receipt_audit, trajectory_audit,
)
from robot.safety.safe_sage_runtime import SafeSAGEControlRuntime
from robot.safety.safe_sage_unicycle_cbf_filter import SafeSAGEUnicycleCBFFilter


def saved_request():
    mpc = planner(0)
    mpc.config = replace(mpc.config, horizon_steps=6)
    control = SafeSAGEControlRuntime(mpc=mpc, backup_filter=SafeSAGEUnicycleCBFFilter(mpc.paper_filter))
    field = runtime_field()
    state = np.array([-2., 2., 0.])
    nominal, goal = nominal_for(state, 6, mpc.config.time_step)
    result = control.plan(state, nominal, field, nominal_goal_grid_xy=goal)
    assert result.success
    return dict(control=control, request=(state, nominal, field, None, goal), expected_result=result)


def save(root, saved):
    root.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps(saved)
    (root / 'window-000.pickle').write_bytes(blob)
    metadata = dict(file='window-000.pickle', sha256=hashlib.sha256(blob).hexdigest(),
                    sensor_timestamp_seconds=saved['request'][2].sensor_timestamp_seconds)
    path = root / 'window-000.json'
    path.write_text(json.dumps(metadata))
    return path


def test_hash_and_sibling_validation_precedes_unpickle(tmp_path):
    path = save(tmp_path, {'request': (None, None, runtime_field())})
    meta = json.loads(path.read_text())
    meta['file'] = '../outside.pickle'
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='sibling'):
        load(path)
    meta['file'] = 'window-000.pickle'
    path.write_text(json.dumps(meta))
    (tmp_path / meta['file']).write_bytes(b'not the captured pickle')
    with pytest.raises(ValueError, match='hash mismatch'):
        load(path)


def test_matrix_preserves_native_parity_and_all_other_configuration(tmp_path):
    saved = saved_request()
    path = save(tmp_path, saved)
    blob, meta = load(path)
    result = probe(blob, meta)
    assert result['baseline']['parity']
    assert [(x['steps'], x['guidance']) for x in result['variants']] == [
        (6, False), (6, True), (30, False), (30, True)]
    expected = asdict(saved['control'].mpc.config)
    for row in result['variants']:
        actual = row['configuration'].copy()
        assert actual.pop('horizon_steps') == row['steps']
        assert actual == {k: v for k, v in expected.items() if k != 'horizon_steps'}
        np.testing.assert_allclose(row['nominal'][:6], saved['request'][1], atol=1e-12, rtol=0)
        assert row['cold_start']
    assert (tmp_path / meta['file']).read_bytes() == blob
    altered = saved.copy()
    altered['request'] = (saved['request'][0], np.zeros((6, 2)), *saved['request'][2:])
    with pytest.raises(AssertionError):
        probe(pickle.dumps(altered), meta)


@pytest.mark.parametrize('changed', [False, True])
def test_receipt_audit_marks_changed_field_as_counterfactual(tmp_path, changed):
    saved = saved_request()
    save(tmp_path / 'planner-replays', saved)
    state, _, field, _, _ = saved['request']
    current = state + (.01, 0., 0.)
    (tmp_path / 'trajectory.json').write_text(json.dumps([state.tolist(), current.tolist()]))
    (tmp_path / 'commands.json').write_text(json.dumps([
        dict(time=0., stop_reason=None), dict(time=.1, stop_reason=None)]))
    (tmp_path / 'report.json').write_text(json.dumps(dict(plan_events=[dict(time=.1,
        guidance_tracking_ablation=dict(initial_chassis_state_grid=state.tolist()))])))
    fields = [dict(sensor_frame_id=field.sensor_frame_id,
        sensor_timestamp=field.sensor_timestamp_seconds, commit_simulation_time=10., state_at_commit=state.tolist())]
    if changed:
        fields.append(dict(sensor_frame_id=field.sensor_frame_id + 1,
            sensor_timestamp=field.sensor_timestamp_seconds + .1, commit_simulation_time=10.1,
            state_at_commit=current.tolist()))
    (tmp_path / 'live-fields.jsonl').write_text('\n'.join(json.dumps(x) for x in fields))
    row = receipt_audit(tmp_path)['rows'][0]
    assert row['same_field_at_receipt'] is (not changed)
    assert row['state_drift_m'] == pytest.approx(.01)
    assert row['original']['accepted']


@pytest.mark.parametrize('value', [np.nan, -1., 2.])
def test_trajectory_diagnostic_never_clips_invalid_command(value):
    saved = saved_request()
    control = saved['control']
    state, _, field, _, _ = saved['request']
    commands = saved['expected_result'].command_sequence.copy()
    commands[0, 0] = value
    with pytest.raises(ValueError, match='silently clip'):
        trajectory_audit(control.mpc, state, commands, field, control)


def test_submillimetre_drift_can_invalidate_dense_tail_with_positive_cbf():
    """Small deterministic reproduction of the native request/receipt failure.

    This preserves the checker: a positive sampled CBF is not substituted for
    the dense padded geometry gate, even when the immediate command is safe.
    """
    from test_safe_sage_cbf_filter import make_fields
    from test_safe_sage_unicycle_mpc import make_mpc

    cs, h, guidance = make_fields(unsafe_yaw_indices=(2,), guidance=(1., 0.))
    mpc = make_mpc(horizon_steps=6, safety_substeps_per_interval=12,
        integration_substeps_per_interval=12, configuration_position_padding_m=.025)
    control = SafeSAGEControlRuntime(mpc=mpc, backup_filter=SafeSAGEUnicycleCBFFilter(mpc.paper_filter))
    original_field = runtime_field()
    field = replace(original_field, pipeline=replace(original_field.pipeline,
        configuration_space=cs, h_full=h, guidance_fields=guidance), temporal_safety_derivative=0.)
    commands = np.tile((.1, 0.), (6, 1))
    state = np.array([-.7451, 0., 0.])
    before = pickle.dumps(field)
    requested = trajectory_audit(mpc, state, commands, field, control)
    received = trajectory_audit(mpc, state + (.0002, 0., 0.), commands, field, control)
    assert requested['accepted']
    assert not received['accepted']
    assert received['dense_unsafe_indices'] == [6]
    assert received['endpoint_unsafe_indices'] == []
    assert received['dtcbf_violation_indices'] == []
    assert received['analytical']['unchanged'] and received['analytical']['satisfied']
    assert pickle.dumps(field) == before
