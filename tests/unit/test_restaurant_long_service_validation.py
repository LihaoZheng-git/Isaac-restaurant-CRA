import hashlib
import json

import pytest

from robot.evaluation.restaurant_long_service_validation import verify_near_human_gates


def test_remote_start_cannot_replace_current_source_near_human_gates(tmp_path):
    (tmp_path/'robot').mkdir()
    source = tmp_path/'robot/controller.py'
    source.write_text('control')
    scene = tmp_path/'scene.usda'
    scene.write_text('scene')
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    folders = []
    for case in ('long-parallel-component', 'long-parallel-near'):
        folder = tmp_path/case
        folder.mkdir()
        folders.append(folder)
        (folder/'invocation.json').write_text(json.dumps(dict(case=case,
            source_sha256={'robot/controller.py': sha(source)}, scene_sha256={str(scene): sha(scene)})))
        (folder/'execution.json').write_text(json.dumps(dict(passed=True, source_unchanged=True)))
        if case.endswith('component'):
            native = folder/'native'
            native.mkdir()
            for name in ('validation', 'human-sensor-validation'):
                (native/(name+'.json')).write_text(json.dumps(dict(passed=True)))
            (native/'robot-setup-validation.json').write_text(json.dumps(dict(
                configured_clearance_m=.5, settled_clearance_m=.5)))
        else:
            (folder/'stationary-gate.json').write_text(json.dumps(dict(passed=True,
                human_component_checks={'ownership': True}, human_body_envelope_checks={'contained': True})))
    assert verify_near_human_gates(folders, tmp_path, scene)
    source.write_text('changed control')
    with pytest.raises(ValueError, match='stale'):
        verify_near_human_gates(folders, tmp_path, scene)
    source.write_text('control')
    gate = folders[1]/'stationary-gate.json'
    gate.write_text(json.dumps(dict(passed=True, human_component_checks={'ownership': True})))
    with pytest.raises(ValueError, match='near-view'):
        verify_near_human_gates(folders, tmp_path, scene)
