from dataclasses import replace
from types import SimpleNamespace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from robot.perception.unknown_obstacle_memory import UnknownObstacleConfig, UnknownObstacleMemory
from robot.perception.verified_dynamic_evidence import verified_human_associations
from robot.evaluation.unknown_obstacle_validation import summarize_unknown_obstacles
from test_unclassified_static_prior import POINTS
from test_stationary_furniture_memory import observe, track
from test_persistent_background_prior import proof_record
from test_confirmed_background_motion import SHAPE, feed


def memory():
    return UnknownObstacleMemory(UnknownObstacleConfig(supported_background_prior=True,
        persistent_background_prior=True, unclassified_static_prior=True,
        require_verified_dynamic_evidence=True))


def association(now=1., key=1, points=POINTS):
    tracking = SimpleNamespace(timestamp_seconds=now, tracked_obstacles=(track(key, points),))
    body = dict(track_id=key, lidar_timestamp_seconds=now, current_support_count=len(points),
        identity_age_seconds=0., original_person_camera_timestamp=now,
        original_person_camera_frame_id=10, identity_basis='fresh_accepted_person_SAM_association')
    return verified_human_associations(tracking, [body], ())


@pytest.mark.parametrize('cue', ['raw_mask', 'uncertain_region'])
@pytest.mark.parametrize('when', ['current', 'lost'])
def test_unverified_cue_cannot_create_permanent_dynamic_label(cue, when):
    m = memory()
    kwargs = dict(excluded_points=POINTS) if cue == 'raw_mask' else dict(human_regions=[(POINTS.mean(axis=0), 10.)])
    observe(m, 1., points=POINTS, **(kwargs if when == 'current' else {}))
    observe(m, 1.2, points=np.empty((0, 2)), **(kwargs if when == 'lost' else {}))
    s = observe(m, 20., points=np.empty((0, 2)))
    r, = s.regions
    assert r.mode == 'unclassified_static_geometry_prior' and r.radius_m == .30
    assert r.original.dynamic_evidence is None and r.original.stationary_support_time is None
    assert r.original.source_time == 1. and r.original.guidance.risk == 1.
    np.testing.assert_array_equal(r.original.points, POINTS)
    assert summarize_unknown_obstacles([proof_record(s)])['passed']


def test_raw_cue_still_excludes_new_stationary_learning():
    m = memory()
    for t in (1., 1.1, 1.2):
        observe(m, t, points=POINTS, excluded_points=POINTS)
    assert m.records[1].background_support is None
    assert not any(len(p) for _, p in m.history)


def test_verified_human_stays_dynamic_across_cue_loss_and_geometry_replacement():
    m = memory()
    observe(m, 1., points=POINTS, verified_human_associations=association())
    observe(m, 1.2, points=POINTS, key=2)
    s = observe(m, 1.4, points=np.empty((0, 2)))
    r, = s.regions
    assert r.mode == 'unknown_motion_reachable_surface_prior' and r.radius_m >= 2.
    e = r.original.dynamic_evidence
    assert e['kind'] == 'verified_human_association' and e['source_time'] == 1.
    assert e['human_association']['original_person_camera_timestamp'] == 1.
    assert summarize_unknown_obstacles([proof_record(s)])['passed']


def test_confirmed_motion_still_creates_dynamic_label():
    m = memory()
    for t, shift in ((1., 0.), (1.2, .08), (1.4, .16)):
        feed(m, t, SHAPE+[1.+shift, 0.])
    s = observe(m, 1.6, points=np.empty((0, 2)))
    assert s.regions[0].original.dynamic_evidence['kind'] == 'confirmed_geometric_motion'
    assert summarize_unknown_obstacles([proof_record(s)])['passed']


@pytest.mark.parametrize('tamper', ['source', 'id', 'count', 'duplicate'])
def test_stale_or_unjoined_human_proof_rejects(tamper):
    p = list(association())
    if tamper == 'source':
        p[0]['source_time'] = .9
    elif tamper == 'id':
        p[0]['track_id'] = 99
    elif tamper == 'count':
        p[0]['current_support_count'] = 3
    else:
        p.append(dict(p[0]))
    with pytest.raises(ValueError):
        observe(memory(), 1., points=POINTS, verified_human_associations=p)


def test_visual_part_requires_current_validated_parent_and_original_epoch():
    parent, child = track(1, POINTS), track(2, POINTS[:3])
    tracking = SimpleNamespace(timestamp_seconds=1., tracked_obstacles=(parent, child))
    body = dict(track_id=1, lidar_timestamp_seconds=1., current_support_count=4,
        identity_age_seconds=0., original_person_camera_timestamp=1.,
        original_person_camera_frame_id=10, identity_basis='fresh_accepted_person_SAM_association')
    part = dict(child_track_id=2, parent_track_id=1, source_timestamp_seconds=1.,
        child_positive_count=3, parent_positive_count=4,
        original_person_camera_timestamp=1., original_person_camera_frame_id=10)
    proofs = verified_human_associations(tracking, [body], [part])
    assert proofs[1]['basis'] == 'validated_current_visual_part'
    part['original_person_camera_timestamp'] = .9
    with pytest.raises(ValueError):
        verified_human_associations(tracking, [body], [part])
    with pytest.raises(ValueError):
        verified_human_associations(tracking, [], [part])


def test_audit_rejects_raw_label_under_verified_policy():
    m = memory()
    observe(m, 1., points=POINTS)
    s = observe(m, 2., points=np.empty((0, 2)))
    record = proof_record(s)
    record['unknown_obstacle_memory']['observed_dynamic_evidence'] = [
        dict(track_id=1, source_time=2., kind='positive_human_region')]
    assert not summarize_unknown_obstacles([record])['passed']


def test_verified_mode_requires_static_default():
    with pytest.raises(ValueError):
        UnknownObstacleConfig(require_verified_dynamic_evidence=True)
    with pytest.raises(ValueError):
        replace(memory().config, require_verified_dynamic_evidence=1)


@pytest.mark.parametrize('case', ['long-parallel', 'long-parallel-near', 'long-parallel-component'])
def test_native_recipe_and_real_cli(case, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    old = json.loads((root/'configs/restaurant_service_demo_v2_compute_paced.json').read_text())['commands'][case]
    command = json.loads((root/'configs/restaurant_service_demo_v2_verified_dynamic.json').read_text())['commands'][case]
    assert command == [*old, '--unknown-verified-dynamic-evidence']
    source = root/command[0]
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(sys, 'argv', [*command, '--output', str(tmp_path/'native')])
    code = compile(source.read_text().split('args.output.mkdir', 1)[0], str(source), 'exec')
    scope = dict(__file__=str(source), __name__='cli_test')
    exec(code, scope)
    assert scope['args'].unknown_verified_dynamic_evidence
    monkeypatch.setattr(sys, 'argv', [c for c in sys.argv if c != '--unknown-unclassified-static-prior'])
    with pytest.raises(SystemExit):
        exec(code, dict(__file__=str(source), __name__='cli_test'))
    assert not (tmp_path/'native').exists()

