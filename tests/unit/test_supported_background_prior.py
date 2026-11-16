from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from robot.perception.unknown_obstacle_memory import UnknownObstacleConfig, UnknownObstacleMemory
from robot.evaluation.unknown_obstacle_validation import summarize_unknown_obstacles
from test_unknown_obstacle_memory import observe, raster
from test_stationary_furniture_memory import GUIDANCE, track


BASE = np.column_stack((np.ones(99), np.linspace(-.1, .1, 99)))
PARTIAL = np.vstack((BASE, [1.18, 0.]))


def memory():
    return UnknownObstacleMemory(UnknownObstacleConfig(supported_background_prior=True))


def test_stronger_flux_is_retained_independently_of_risk_ordering():
    from robot.safety.class_safety_policy import ClassSafetyParameters
    old = replace(GUIDANCE, safety_parameters=ClassSafetyParameters(.7, 2.))
    new = replace(GUIDANCE, safety_parameters=ClassSafetyParameters(.9, 1.6))
    m = memory()
    for t, g in ((1., old), (1.1, new)):
        m.update(BASE, np.ones(len(BASE), dtype=np.int64), (track(1, BASE),), {1: g},
            source_time=t, source_frame=round(t*10), robot_xy=(0., 0.))
    assert m.records[1].guidance.risk == .9
    assert m.records[1].guidance.boundary_flux_magnitude == 2.


def acquire(m, points=PARTIAL):
    observe(m, 1., points=BASE)
    observe(m, 1.1, points=BASE)
    observe(m, 1.2, points=points)


def test_partial_support_keeps_whole_surface_and_separate_original_clocks():
    m = memory()
    acquire(m)
    s = observe(m, 1.4, key=2, points=BASE)
    r, = s.regions
    assert r.mode == 'supported_background_position_prior'
    assert r.original.source_time == r.original.stationary_qualified_time == 1.2
    assert r.original.source_frame == 12
    assert r.original.stationary_support_time == 1.4
    np.testing.assert_array_equal(r.original.points, PARTIAL)
    support = r.original.background_support
    assert support['supported_point_count'] == 99
    assert support['unsupported_point_count'] == 1
    assert support['supported_fraction'] == .99
    assert support['maximum_gap_m'] == pytest.approx(.18)
    assert r.radius_m == .25 and r.original.guidance.risk == GUIDANCE.risk
    proof = s.coverage(1, 1.2, PARTIAL.mean(0), raster(s), (0., 0.), 0., original_points=PARTIAL)
    assert proof and proof['background_support'] == support
    report = summarize_unknown_obstacles([dict(unknown_obstacle_memory=s.diagnostic(),
        unknown_obstacle_field_checks=[proof])])
    assert report['passed']


def test_default_off_still_treats_same_partial_surface_as_dynamic_unknown():
    m = UnknownObstacleMemory()
    acquire(m)
    s = observe(m, 1.4, key=2, points=BASE)
    assert s.regions[0].mode == 'unknown_motion_reachable_surface_prior'


@pytest.mark.parametrize('points', [np.vstack((BASE, [1.3, 0.])),
    np.vstack((BASE, [1.18, 0.], [1.18, .1]))])
def test_large_missing_tail_or_low_fraction_does_not_establish_stationary_prior(points):
    m = memory()
    acquire(m, points)
    assert m.records[1].stationary_support_time is None
    s = observe(m, 1.4, key=2, points=BASE)
    assert s.regions[0].mode == 'unknown_motion_reachable_surface_prior'


def test_partial_support_is_retained_for_bounded_lease_then_dynamic_growth():
    m = memory()
    acquire(m)
    s = observe(m, 1.4, points=np.empty((0, 2)))
    r, = s.regions
    assert r.mode == 'retained_supported_background_prior'
    assert r.radius_m == pytest.approx(.251)
    assert r.original.stationary_support_time == 1.2
    s = observe(m, 12., points=np.empty((0, 2)))
    r, = s.regions
    assert r.mode == 'unknown_motion_reachable_surface_prior'
    assert r.radius_m == pytest.approx(.30+.8+1.55)
    assert r.original.source_time == 1.2


@pytest.mark.parametrize('exclusion', ['mask', 'body'])
def test_human_exclusion_invalidates_whole_background_hypothesis(exclusion):
    m = memory()
    acquire(m)
    args = {'excluded_points': np.array([[1., 0.]])} if exclusion == 'mask' else {
        'human_regions': [(np.array([1., 0.]), .65)]}
    s = observe(m, 1.4, key=2, points=BASE, **args)
    assert s.regions[0].mode == 'unknown_motion_reachable_surface_prior'
    assert s.regions[0].original.background_support is None
    assert all(r.stationary_support_time is None for r in m.records.values())


def test_person_excluded_history_cannot_later_qualify_without_three_new_sources():
    m = memory()
    for t in (1., 1.1, 1.2):
        observe(m, t, points=PARTIAL, excluded_points=[[1., 0.]])
    observe(m, 1.4, points=PARTIAL)
    observe(m, 1.5, points=PARTIAL)
    assert m.records[1].stationary_support_time is None
    observe(m, 1.6, points=PARTIAL)
    assert m.records[1].stationary_support_time == 1.6


def test_motion_without_fixed_world_support_never_qualifies():
    m = memory()
    for t, x in ((1., 0.), (1.1, .15), (1.2, .3)):
        observe(m, t, points=BASE+[x, 0.])
    assert m.records[1].stationary_support_time is None


def test_current_observed_mover_invalidates_overlapping_background_prior():
    m = memory()
    acquire(m)
    moving = BASE + [.18, 0.]
    p = np.vstack((BASE, moving))
    ids = np.r_[np.full(len(BASE), 2), np.full(len(moving), 3)]
    s = m.update(p, ids, (track(2, BASE), track(3, moving, (.5, 0.))),
        {2: GUIDANCE, 3: GUIDANCE}, source_time=1.4, source_frame=14, robot_xy=(0., 0.))
    assert s.regions[0].mode == 'unknown_motion_reachable_surface_prior'
    assert s.regions[0].original.stationary_support_time is None


def test_source_gap_clears_partial_stationary_candidates():
    m = memory()
    observe(m, 1., points=BASE)
    observe(m, 1.1, points=BASE)
    observe(m, 1.5, points=PARTIAL)
    assert m.records[1].stationary_support_time is None
    observe(m, 1.6, points=PARTIAL)
    assert m.records[1].stationary_support_time is None
    observe(m, 1.7, points=PARTIAL)
    assert m.records[1].stationary_support_time == 1.7


def test_duplicate_returns_cannot_fake_three_distinct_support_points():
    m = memory()
    for t in (1., 1.1, 1.2):
        observe(m, t, points=np.tile([[1., 0.]], (100, 1)))
    assert m.records[1].stationary_support_time is None


def test_three_samples_in_too_short_window_cannot_qualify():
    m = memory()
    for t in (1., 1.05, 1.1):
        observe(m, t, points=PARTIAL)
    assert m.records[1].stationary_support_time is None


def test_current_geometry_refresh_does_not_renew_missing_original_factor_source():
    m = memory()
    acquire(m)
    for t in np.arange(1.4, 12., .2):
        s = observe(m, float(t), key=2, points=BASE)
    r, = s.regions
    assert r.original.source_time == 1.2
    assert r.original.stationary_support_time == s.source_time
    assert r.mode == 'supported_background_position_prior'
    assert r.radius_m == .25


def test_dynamic_conflict_uses_last_geometric_support_without_renewing_factor_epoch():
    m = memory()
    acquire(m)
    for t in np.arange(1.4, 12., .2):
        s = observe(m, float(t), key=2, points=BASE)
    last_support = s.source_time
    s = observe(m, 12., points=np.empty((0, 2)), human_regions=[(np.array([1., 0.]), .65)])
    r = next(r for r in s.regions if r.original.track_id == 1)
    assert r.mode == 'unknown_motion_reachable_surface_prior'
    assert r.original.source_time == 1.2
    assert r.original.motion_support_time == last_support
    assert r.original.stationary_support_time is None
    assert r.radius_m == pytest.approx(.25 + 12.-last_support + 1.55)


@pytest.mark.parametrize('kwargs', [dict(supported_background_prior=1),
    dict(background_support_fraction=.98), dict(background_maximum_gap_m=.26),
    dict(background_position_radius_m=.1)])
def test_invalid_or_undeclared_bounds_reject(kwargs):
    with pytest.raises(ValueError):
        UnknownObstacleConfig(**kwargs)


@pytest.mark.parametrize('case', ['long-parallel', 'long-parallel-near', 'long-parallel-component'])
def test_real_native_recipe_explicit_opt_in_and_dependency(case, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    old = json.loads((root/'configs/restaurant_service_demo_v2_unknown_obstacles.json').read_text())['commands'][case]
    command = json.loads((root/'configs/restaurant_service_demo_v2_supported_background.json').read_text())['commands'][case]
    assert command == [*old, '--unknown-supported-background-prior']
    source = root/command[0]
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(sys, 'argv', [*command, '--output', str(tmp_path/'native')])
    scope = dict(__file__=str(source), __name__='cli_test')
    code = compile(source.read_text().split('args.output.mkdir', 1)[0], str(source), 'exec')
    exec(code, scope)
    assert scope['args'].unknown_supported_background_prior
    monkeypatch.setattr(sys, 'argv', [c for c in sys.argv if c != '--unknown-obstacle-planning'])
    with pytest.raises(SystemExit):
        exec(code, dict(__file__=str(source), __name__='cli_test'))
    assert not (tmp_path/'native').exists()


def test_worker_staged_support_history_rollback():
    from test_object_risk_map import mapped_worker
    from test_live_object_perception import packet
    from robot.evaluation.live_checked_wait import StagedLivePerception
    from robot.safety.rolling_local_field import RollingLocalField
    w = mapped_worker()
    w.rolling_field = RollingLocalField(w.grid_config)
    w.unknown_obstacle_memory = memory()
    for frame, t in ((1, 1.), (2, 1.1)):
        w(replace(packet(frame, t)))
    staged = StagedLivePerception(w)
    result = staged(replace(packet(3, 1.2)))
    assert w.unknown_obstacle_memory.last_time == 1.1
    staged.discard()
    assert len(w.unknown_obstacle_memory.history) == 2
    result = staged(replace(packet(4, 1.3)))
    staged.accept(result)
    assert w.unknown_obstacle_memory.last_time == 1.3
    assert any(r.background_support for r in w.unknown_obstacle_memory.records.values())
