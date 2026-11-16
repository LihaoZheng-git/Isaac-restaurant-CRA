import json
from pathlib import Path
import sys

import numpy as np
import pytest

from robot.scene.restaurant_service_build import LONG_ROUTE_WORLD_XY, restaurant_route_local_xy


def test_wide_corner_changes_only_one_authored_guide_and_preserves_goal():
    legacy = np.array(restaurant_route_local_xy('long-service'))
    wide = np.array(restaurant_route_local_xy('long-service', wide_corner=True))
    assert legacy == pytest.approx(np.array(LONG_ROUTE_WORLD_XY[1:])-LONG_ROUTE_WORLD_XY[0])
    difference = wide-legacy
    expected = np.zeros_like(difference)
    expected[2, 0] = .4
    assert difference == pytest.approx(expected)
    assert tuple(wide[-1]) == tuple(legacy[-1])


@pytest.mark.parametrize('name,enabled', [('short',True),('service',True),('human-service',True),('long-service','yes')])
def test_wide_corner_cannot_change_other_routes(name, enabled):
    with pytest.raises(ValueError, match='explicit long-service'):
        restaurant_route_local_xy(name,wide_corner=enabled)


def test_wide_recipe_and_real_parser_preserve_all_other_experiments(monkeypatch,tmp_path):
    root=Path(__file__).resolve().parents[2]
    before=json.loads((root/'configs/restaurant_service_demo_v2_upper_departure.json').read_text())['commands']
    after=json.loads((root/'configs/restaurant_service_demo_v2_wide_corner.json').read_text())['commands']
    for key,command in after.items():
        assert command == before[key]+(['--service-wide-corner','--save-failed-planner-replay'] if key=='long-parallel' else [])
    source=root/after['long-parallel'][0]
    code=compile(source.read_text().split('args.output.mkdir',1)[0],str(source),'exec')
    monkeypatch.setattr(sys,'path',list(sys.path))
    monkeypatch.setattr(sys,'argv',[*after['long-parallel'],'--output',str(tmp_path/'native')])
    scope=dict(__file__=str(source),__name__='cli_test')
    exec(code,scope)
    assert scope['args'].service_wide_corner
    assert scope['args'].save_failed_planner_replay
    monkeypatch.setattr(sys,'argv',[x for x in sys.argv if x!='--service-demo-v2'])
    with pytest.raises(SystemExit):
        exec(code,dict(__file__=str(source),__name__='cli_test'))
    assert not (tmp_path/'native').exists()
