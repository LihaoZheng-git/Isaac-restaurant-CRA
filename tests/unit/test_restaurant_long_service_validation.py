import hashlib
import json

import pytest

from robot.evaluation.restaurant_long_service_validation import verify_near_human_gates


def test_distant_start_defers_exercise_count_but_never_provenance_failure():
    from robot.evaluation.restaurant_long_service_validation import distant_start_retention_audit
    report = dict(passed=False, checks=dict(original_fresh_visual_prior=True,
        current_positive_body_identity=True, original_source_expiry=True,
        decayed_confidence_widened_bounds=True, five_distinct_retained_sources=False))
    assert distant_start_retention_audit(report)['passed']
    assert not distant_start_retention_audit(report)['full_retention_gate_passed']
    for key in list(report['checks'])[:-1]:
        bad = dict(report, checks=dict(report['checks'], **{key: False}))
        assert not distant_start_retention_audit(bad)['passed']


def test_remote_start_cannot_replace_current_source_near_human_gates(tmp_path):
    (tmp_path/'robot').mkdir()
    source = tmp_path/'robot/controller.py'
    source.write_text('control')
    scene = tmp_path/'scene.usda'
    scene.write_text('scene')
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    folders = []
    for case in ('long-parallel-component', 'long-parallel-near', 'long-idle-component'):
        folder = tmp_path/case
        folder.mkdir()
        folders.append(folder)
        (folder/'invocation.json').write_text(json.dumps(dict(case=case,
            source_sha256={'robot/controller.py': sha(source)}, scene_sha256={str(scene): sha(scene)})))
        (folder/'execution.json').write_text(json.dumps(dict(passed=True, source_unchanged=True)))
        if case == 'long-idle-component':
            native = folder/'native'
            native.mkdir()
            checks = {k: True for k in ('hold_at_least_180_seconds', 'held_agent_disabled',
                'idle_drift_below_1mm', 'released_native_walking', 'native_task_completed',
                'native_travel_over_3m', 'unchanged_geometry_guard')}
            (native/'validation.json').write_text(json.dumps(dict(passed=True, checks=checks)))
        elif case.endswith('component'):
            native = folder/'native'
            native.mkdir()
            for name in ('validation', 'human-sensor-validation'):
                (native/(name+'.json')).write_text(json.dumps(dict(passed=True)))
            (native/'robot-setup-validation.json').write_text(json.dumps(dict(
                configured_clearance_m=.5, settled_clearance_m=.5)))
        else:
            (folder/'stationary-gate.json').write_text(json.dumps(dict(passed=True,
                human_component_checks={'ownership': True}, human_body_envelope_checks={'contained': True})))
    with pytest.raises(ValueError, match='three'):
        verify_near_human_gates(folders[:2], tmp_path, scene)
    assert verify_near_human_gates(folders, tmp_path, scene)
    idle_gate = folders[2]/'native/validation.json'
    idle_report = json.loads(idle_gate.read_text())
    idle_report['checks']['idle_drift_below_1mm'] = False
    idle_gate.write_text(json.dumps(idle_report))
    with pytest.raises(ValueError, match='long idle'):
        verify_near_human_gates(folders, tmp_path, scene)
    idle_report['checks']['idle_drift_below_1mm'] = True
    idle_gate.write_text(json.dumps(idle_report))
    source.write_text('changed control')
    with pytest.raises(ValueError, match='stale'):
        verify_near_human_gates(folders, tmp_path, scene)
    source.write_text('control')
    gate = folders[1]/'stationary-gate.json'
    gate.write_text(json.dumps(dict(passed=True, human_component_checks={'ownership': True})))
    with pytest.raises(ValueError, match='near-view'):
        verify_near_human_gates(folders, tmp_path, scene)
