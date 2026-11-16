"""Scene removals and diagnostic poses must not alter runtime controller inputs."""
import json
from pathlib import Path

from pxr import Usd, UsdGeom
import pytest

from robot.scene.restaurant_demo_build import DEMO_ROOT, author_demo_props, demo_prop_records
from robot.scene.restaurant_service_build import SERVICE_ROOT, active_service_props, service_props
from robot.scene.restaurant_service_v3 import BALANCED_CONTACT_CENTERS
from test_restaurant_demo_build import _stage


def test_removed_dressing_has_no_independent_obstacle_box():
    stage = _stage()
    author_demo_props(stage)
    for name in ('ClearingStation', 'ClosedCaddy', 'UtensilCaddy'):
        stage.GetPrimAtPath(DEMO_ROOT+'/'+name).SetActive(False)
    assert {r['name'] for r in demo_prop_records(stage)} == {'RoundedBin', 'PulledChair'}
    for spec in service_props():
        UsdGeom.Xform.Define(stage, SERVICE_ROOT+'/'+spec.name)
    stage.GetPrimAtPath(SERVICE_ROOT+'/SecondPot').SetActive(False)
    assert 'SecondPot' not in {p.name for p in active_service_props(stage)}
    with pytest.raises(ValueError, match='missing restaurant dressing'):
        active_service_props(Usd.Stage.CreateInMemory())


@pytest.mark.parametrize('variant',['separated','straight','open_aisle'])
def test_separated_mission_changes_only_the_scene_reference(variant):
    old=json.loads(Path('configs/restaurant_service_demo_v3.json').read_text())['commands']
    new=json.loads(Path(f'configs/restaurant_service_demo_v3_{variant}.json').read_text())['commands']
    for case, original in old.items():
        changed=new[case].copy()
        key='--scene' if '--scene' in changed else '--restaurant-scene'
        index=changed.index(key)+1
        changed[index]=original[index]
        assert changed == original


@pytest.mark.parametrize('case',['bin-static','crate-static'])
@pytest.mark.parametrize('variant',['separated','straight','open_aisle'])
def test_explicit_prop_pose_is_zero_wheel_only(monkeypatch,case,variant):
    path=Path('scripts/demos/cra_live_object_isaac_demo.py').resolve()
    cmd=json.loads(Path(f'configs/restaurant_service_demo_v3_{variant}.json').read_text())['commands'][case]
    cmd+=['--output','outputs/unused-separated-cli']
    monkeypatch.setattr('sys.argv',cmd)
    code=compile(path.read_text().split('args.output.mkdir',1)[0],str(path),'exec')
    namespace={'__file__':str(path)}
    exec(code,namespace)
    assert namespace['args'].stationary_prop_world_pose[1:] == [4.05,0.]
    cmd.append('--motion')
    with pytest.raises(SystemExit):
        exec(code,{'__file__':str(path)})


def test_balanced_service_preserves_the_corrected_contact_profile_and_separates_props():
    previous = json.loads(Path('configs/restaurant_service_demo_v3_corrected_contact.json').read_text())['commands']
    current = json.loads(Path('configs/restaurant_service_demo_v3_balanced_contact.json').read_text())['commands']
    assert previous.keys() == current.keys()
    for case, original in previous.items():
        command = current[case].copy()
        if case in ('long-parallel', 'long-parallel-near'):
            assert command.pop() == '--observe-service-clearance'
        scene_index = command.index('--scene' if '--scene' in command else '--restaurant-scene') + 1
        assert command[scene_index] == original[scene_index].replace('_open_aisle_', '_balanced_')
        command[scene_index] = original[scene_index]
        assert command == original
    bin_center, crate_center = BALANCED_CONTACT_CENTERS
    assert bin_center[0] - crate_center[0] == pytest.approx(3.7)
    assert 4.05 - bin_center[1] == pytest.approx(.65)
    assert crate_center == (17.5, bin_center[1])
