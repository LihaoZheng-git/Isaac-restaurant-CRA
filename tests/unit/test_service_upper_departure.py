import json
from pathlib import Path
import sys

import pytest

from robot.scene.restaurant_background_walk import (
    BackgroundServiceWalk, UPPER_DEPARTURE_ROUTE, audit_background_walk,
)


def test_upper_departure_preserves_encounter_then_stays_out_of_delivery_corridor():
    legacy, upper = BackgroundServiceWalk(), BackgroundServiceWalk(upper_departure=True)
    for i in range(8):
        assert legacy.goal == upper.goal
        legacy.advance(legacy.goal, i, encounter_due=True)
        upper.advance(upper.goal, i, encounter_due=True)
    assert upper.phase == legacy.phase == 'continue_walk'
    assert upper.goal == UPPER_DEPARTURE_ROUTE[2]
    assert upper.goal != legacy.goal
    for i in range(4):
        upper.advance(upper.goal, 8+i)
    for i in range(10):
        assert upper.goal[1] >= 18.5
        upper.advance(upper.goal, 12+i)
    assert not any(e['next_phase'] == 'idle_before' for e in upper.events)


def test_upper_route_cannot_skip_actual_arrival():
    route = BackgroundServiceWalk(upper_departure=True)
    assert not route.advance((29.7,23.5,0), 1, encounter_due=True)
    assert route.phase == 'background_walk'
    with pytest.raises(ValueError):
        BackgroundServiceWalk(upper_departure='yes')


def test_native_actor_domain_covers_entire_upper_route_and_keeps_legacy_bounds():
    from robot.scene.restaurant_human_motion import human_navmesh_extent
    low, high = human_navmesh_extent(parallel_oncoming=True, upper_departure=True)
    for goal in UPPER_DEPARTURE_ROUTE:
        assert all(low[i]+.4 < goal[i] < high[i]-.4 for i in (0,1))
    assert human_navmesh_extent() == ((27.,7.,-.2),(31.,14.,2.5))
    assert human_navmesh_extent(parallel_oncoming=True) == ((25.,6.,-.2),(31.,14.,2.5))
    with pytest.raises(ValueError):
        human_navmesh_extent(upper_departure=True)


def test_upper_actor_audit_requires_explicit_mode_and_actual_upper_entry():
    from robot.scene.restaurant_background_walk import BACKGROUND_ROUTE
    rows = [dict(time=i*.1, actor_xyz=[29.7, 7+i*.05, 0.],
        phase='background_walk' if i<40 else 'move' if i<60 else 'continue_walk',
        actor_layout_clearance_m=.2, collider_alignment_error_m=0.,
        background_upper_departure=True) for i in range(241)]
    events = [dict(previous_phase='background_walk', previous_goal=BACKGROUND_ROUTE[-1]),
        dict(previous_phase='move', next_phase='continue_walk'),
        dict(previous_phase='continue_walk', previous_goal=UPPER_DEPARTURE_ROUTE[3])]
    assert audit_background_walk(rows, events, minimum_background_seconds=3., upper_departure=True)['passed']
    assert not audit_background_walk(rows, events, minimum_background_seconds=3.)['checks']['bounded_region']
    assert not audit_background_walk(rows, events[:-1], minimum_background_seconds=3., upper_departure=True)['checks']['upper_aisle_reached']
    rows[-1]['background_upper_departure'] = False
    assert not audit_background_walk(rows, events, minimum_background_seconds=3., upper_departure=True)['checks']['explicit_upper_departure']
    rows[-1]['background_upper_departure'] = True
    rows[-1]['actor_layout_clearance_m'] = .01
    assert not audit_background_walk(rows, events, minimum_background_seconds=3., upper_departure=True)['checks']['independent_geometry']


@pytest.mark.parametrize('case', ['long-parallel','long-idle-component'])
def test_upper_recipe_is_only_actor_option_and_real_parser_requires_background(case, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    before = json.loads((root/'configs/restaurant_service_demo_v2_part_recovery.json').read_text())['commands']
    after = json.loads((root/'configs/restaurant_service_demo_v2_upper_departure.json').read_text())['commands']
    for key, command in after.items():
        assert command == before[key]+(['--background-upper-departure'] if key in ('long-parallel','long-idle-component') else [])
    source = root/after[case][0]
    code = compile(source.read_text().split('args.output.mkdir',1)[0],str(source),'exec')
    monkeypatch.setattr(sys,'path',list(sys.path))
    monkeypatch.setattr(sys,'argv',[*after[case],'--output',str(tmp_path/'native')])
    scope = dict(__file__=str(source),__name__='cli_test')
    exec(code,scope)
    assert scope['args'].background_upper_departure
    monkeypatch.setattr(sys,'argv',[c for c in sys.argv if c!='--background-service-walk'])
    with pytest.raises(SystemExit):
        exec(code,dict(__file__=str(source),__name__='cli_test'))
    assert not (tmp_path/'native').exists()
