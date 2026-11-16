"""The closer-crate fixture must retain the existing evidence/control gates."""
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('motion', [False, True])
def test_probe_cli_requires_refresh_and_original_start(monkeypatch, motion):
    path = Path('scripts/demos/cra_live_object_isaac_demo.py').resolve()
    command = json.loads(Path('configs/restaurant_crate_route_probe.json').read_text())['commands']['crate-static']
    command += ['--output', 'outputs/unused-crate-route-probe']
    if motion:
        command += ['--motion']
    code = compile(path.read_text().split('args.output.mkdir', 1)[0], str(path), 'exec')
    monkeypatch.setattr('sys.argv', command)
    namespace = {'__file__': str(path)}
    exec(code, namespace)
    assert namespace['args'].crate_route_probe
    for missing in ('--confirmed-prop-refresh', '--static-prop-evidence', '--image-object-approach'):
        changed = command.copy()
        changed.remove(missing)
        monkeypatch.setattr('sys.argv', changed)
        with pytest.raises(SystemExit):
            exec(code, {'__file__': str(path)})
    changed = command.copy()
    changed[changed.index('--stationary-prop-world-pose') + 2] = '4.2'
    monkeypatch.setattr('sys.argv', changed)
    with pytest.raises(SystemExit):
        exec(code, {'__file__': str(path)})


def test_probe_changes_only_scene_route_and_duration():
    original = json.loads(Path('configs/restaurant_crate_evidence_refresh.json').read_text())['commands']['crate-static']
    changed = json.loads(Path('configs/restaurant_crate_route_probe.json').read_text())['commands']['crate-static']
    changed.remove('--crate-route-probe')
    for option in ('--restaurant-scene', '--maximum-seconds'):
        changed[changed.index(option) + 1] = original[original.index(option) + 1]
    assert changed == original
