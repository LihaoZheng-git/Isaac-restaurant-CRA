from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from robot.perception.unknown_obstacle_memory import UnknownObstacleConfig, UnknownObstacleMemory
from robot.evaluation.unknown_obstacle_validation import UnknownObstacleGeometryArchive, summarize_unknown_obstacles
from test_supported_background_prior import acquire, BASE, PARTIAL
from test_unknown_obstacle_memory import observe, raster
from test_stationary_furniture_memory import GUIDANCE, track


def memory():
    return UnknownObstacleMemory(UnknownObstacleConfig(supported_background_prior=True, persistent_background_prior=True))


def proof_record(snapshot):
    grid = raster(snapshot)
    return dict(unknown_obstacle_memory=snapshot.diagnostic(), unknown_obstacle_field_checks=[
        snapshot.coverage(r.original.track_id, r.original.source_time, r.original.center,
            grid, (0., 0.), 0., original_points=r.original.points) for r in snapshot.regions])


@pytest.mark.parametrize('now,mode,radius', [(1.4, 'retained_supported_background_prior', .251),
    (11.3, 'persistent_stationary_background_prior', .30),
    (300., 'persistent_stationary_background_prior', .30)])
def test_persistent_surface_caps_without_renewing_original_or_support_clocks(now, mode, radius):
    m = memory()
    acquire(m)
    original = m.records[1]
    snapshot = observe(m, now, points=np.empty((0, 2)))
    region, = snapshot.regions
    assert region.mode == mode and region.radius_m == pytest.approx(radius)
    np.testing.assert_array_equal(region.original.points, PARTIAL)
    assert region.original.source_time == region.original.stationary_support_time == 1.2
    assert region.original.source_frame == 12
    assert region.original.stationary_qualified_time == original.stationary_qualified_time
    assert region.original.background_support == original.background_support
    assert region.original.guidance == original.guidance
    assert summarize_unknown_obstacles([proof_record(snapshot)])['passed']


def test_unqualified_surface_still_dynamic_despite_persistent_option():
    m = memory()
    observe(m, 1.)
    snapshot = observe(m, 20., points=np.empty((0, 2)))
    region, = snapshot.regions
    assert region.mode == 'unknown_motion_reachable_surface_prior'
    assert region.radius_m == pytest.approx(.1+19+1.55)


@pytest.mark.parametrize('kind', ['mask', 'body', 'mover'])
def test_persistent_hypothesis_ends_on_positive_person_or_motion_contradiction(kind):
    m = memory()
    acquire(m)
    observe(m, 20., points=np.empty((0, 2)))
    if kind == 'mover':
        points = BASE + [.18, 0.]
        snapshot = m.update(points, np.full(len(points), 2), (track(2, points, (.5, 0.)),),
            {2: GUIDANCE}, source_time=20.2, source_frame=202, robot_xy=(0., 0.))
    else:
        kwargs = {'excluded_points': [[1., 0.]]} if kind == 'mask' else {'human_regions': [(np.array([1., 0.]), .65)]}
        snapshot = observe(m, 20.2, points=np.empty((0, 2)), **kwargs)
    region = next(r for r in snapshot.regions if r.original.track_id == 1)
    assert region.mode == 'unknown_motion_reachable_surface_prior'
    assert region.original.background_support is None
    assert region.original.stationary_support_time is None
    assert region.original.motion_support_time == region.original.source_time == 1.2
    assert region.radius_m > .30
    snapshot = observe(m, 20.4, points=np.empty((0, 2)))
    assert next(r for r in snapshot.regions if r.original.track_id == 1).mode == 'unknown_motion_reachable_surface_prior'


def test_fresh_real_support_after_gap_is_distinct_from_persistence():
    m = memory()
    acquire(m)
    observe(m, 20., points=np.empty((0, 2)))
    for t in (20.1, 20.2):
        snapshot = observe(m, t, key=2, points=BASE)
        old = next(r for r in snapshot.regions if r.original.track_id == 1)
        assert old.original.stationary_support_time == 1.2
    snapshot = observe(m, 20.3, key=2, points=BASE)
    old = next(r for r in snapshot.regions if r.original.track_id == 1)
    assert old.original.stationary_support_time == 20.3
    assert old.original.source_time == 1.2
    assert old.radius_m == .25


@pytest.mark.parametrize('kwargs', [dict(persistent_background_prior=True),
    dict(supported_background_prior=True, persistent_background_prior=1),
    dict(supported_background_prior=True, persistent_background_prior=True, background_maximum_radius_m=.31),
    dict(supported_background_prior=True, persistent_background_prior=True, background_maximum_radius_m=.24),
    dict(supported_background_prior=True, persistent_background_prior=True, stationary_radius_m=.4)])
def test_invalid_persistence_dependency_or_cap_rejects(kwargs):
    with pytest.raises(ValueError):
        UnknownObstacleConfig(**kwargs)


@pytest.mark.parametrize('tamper', ['cap', 'opt_in', 'age'])
def test_auditor_rejects_false_persistence_provenance(tamper):
    m = memory()
    acquire(m)
    record = proof_record(observe(m, 20., points=np.empty((0, 2))))
    if tamper == 'cap':
        record['unknown_obstacle_memory']['regions'][0]['radius_m'] = .31
    elif tamper == 'opt_in':
        record['unknown_obstacle_memory']['config']['persistent_background_prior'] = False
    else:
        record['unknown_obstacle_memory']['regions'][0]['stationary_support_time'] = 19.
    assert not summarize_unknown_obstacles([record])['passed']


def test_archive_exact_original_geometry_once_and_reject_changed_existing_payload(tmp_path):
    m = memory()
    acquire(m)
    archive = UnknownObstacleGeometryArchive(tmp_path)
    first = archive.save(observe(m, 2., points=np.empty((0, 2))))
    later = archive.save(observe(m, 20., points=np.empty((0, 2))))
    assert first['regions'] == later['regions']
    ref, = later['regions']
    files = list(tmp_path.rglob('*.npz'))
    assert len(files) == 1 and ref['original_point_count'] == 100
    with np.load(files[0], allow_pickle=False) as data:
        np.testing.assert_array_equal(data['original_points_world_xy'], PARTIAL)
        metadata = str(data['metadata_json'])
        assert json.loads(metadata)['original_source_time'] == 1.2
    np.savez_compressed(files[0], metadata_json=np.array(metadata), original_points_world_xy=BASE)
    with pytest.raises(ValueError, match='archive mismatch'):
        UnknownObstacleGeometryArchive(tmp_path).save(observe(m, 20.2, points=np.empty((0, 2))))


def test_archive_includes_every_region_and_disambiguates_changed_geometry(tmp_path):
    m = memory()
    acquire(m)
    snapshot = observe(m, 20., points=np.empty((0, 2)))
    original = snapshot.regions[0]
    second = replace(original, original=replace(original.original, points=BASE))
    refs = UnknownObstacleGeometryArchive(tmp_path).save(replace(snapshot, regions=(original, second)))['regions']
    assert len(refs) == 2 and refs[0]['path'] != refs[1]['path']
    assert len(list(tmp_path.rglob('*.npz'))) == 2


@pytest.mark.parametrize('case', ['long-parallel', 'long-parallel-near', 'long-parallel-component'])
def test_real_persistent_recipe_has_only_explicit_option_and_dependency(case, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    old = json.loads((root/'configs/restaurant_service_demo_v2_supported_background.json').read_text())['commands'][case]
    command = json.loads((root/'configs/restaurant_service_demo_v2_persistent_background.json').read_text())['commands'][case]
    assert command == [*old, '--unknown-persistent-background-prior']
    source = root/command[0]
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(sys, 'argv', [*command, '--output', str(tmp_path/'native')])
    code = compile(source.read_text().split('args.output.mkdir', 1)[0], str(source), 'exec')
    scope = dict(__file__=str(source), __name__='cli_test')
    exec(code, scope)
    assert scope['args'].unknown_persistent_background_prior
    monkeypatch.setattr(sys, 'argv', [c for c in sys.argv if c != '--unknown-supported-background-prior'])
    with pytest.raises(SystemExit):
        exec(code, dict(__file__=str(source), __name__='cli_test'))
    assert not (tmp_path/'native').exists()
