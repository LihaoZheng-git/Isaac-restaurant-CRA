from dataclasses import replace

import numpy as np
import pytest

from robot.perception.unknown_obstacle_memory import UnknownObstacleConfig, UnknownObstacleMemory
from robot.evaluation.unknown_obstacle_validation import summarize_unknown_obstacles
from test_supported_background_prior import acquire, BASE, PARTIAL
from test_persistent_background_prior import memory, proof_record
from test_stationary_furniture_memory import GUIDANCE, track


def merged_update(m, tail, now=1.4, base=BASE, **kwargs):
    points = np.vstack((base, tail))
    return m.update(points, np.full(len(points), 2), (track(2, points, (.5, 0.)),),
        {2: GUIDANCE}, source_time=now, source_frame=round(now*10), robot_xy=(0., 0.), **kwargs)


FAR_TAIL = np.array([[1.5, 0.], [1.6, 0.]])


def test_moving_centroid_cannot_invalidate_qualified_background_through_static_points():
    m = memory()
    acquire(m)
    snapshot = merged_update(m, FAR_TAIL)
    old, = snapshot.regions
    assert old.mode == 'supported_background_position_prior'
    assert old.original.source_time == 1.2
    np.testing.assert_array_equal(old.original.points, PARTIAL)
    proof, = snapshot.background_motion_proofs
    assert proof['repeated_stationary_point_count'] == 99
    assert proof['unconfirmed_motion_point_count'] == 2
    assert proof['minimum_unconfirmed_distance_m'] == pytest.approx(.32)
    assert proof['stationary_overlap_only']
    assert m.records[2].background_support is None  # New mixed cluster is NOT qualified.
    assert summarize_unknown_obstacles([proof_record(snapshot)])['passed']


def test_new_unconfirmed_motion_near_old_surface_still_invalidates_whole_prior():
    m = memory()
    acquire(m)
    snapshot = merged_update(m, [[1.18, 0.], [1.5, 0.]], base=BASE[:20])
    old, = snapshot.regions
    assert old.mode == 'unknown_motion_reachable_surface_prior'
    assert old.original.background_support is None
    proof, = snapshot.background_motion_proofs
    assert not proof['stationary_overlap_only']
    assert proof['minimum_unconfirmed_distance_m'] == 0.
    assert summarize_unknown_obstacles([proof_record(snapshot)])['passed']


@pytest.mark.parametrize('kind', ['mask', 'body'])
def test_human_exclusions_still_override_repeated_stationary_overlap(kind):
    m = memory()
    acquire(m)
    kwargs = {'excluded_points': [[1., 0.]]} if kind == 'mask' else {'human_regions': [(np.array([1., 0.]), .65)]}
    snapshot = merged_update(m, FAR_TAIL, **kwargs)
    assert snapshot.regions[0].mode == 'unknown_motion_reachable_surface_prior'


def test_source_gap_cannot_supply_stationary_point_exemption():
    m = memory()
    acquire(m)
    snapshot = merged_update(m, FAR_TAIL, now=2.)
    assert snapshot.regions[0].mode == 'unknown_motion_reachable_surface_prior'
    proof, = snapshot.background_motion_proofs
    assert proof['repeated_stationary_point_count'] == 0
    assert not proof['stationary_overlap_only']


def test_legacy_supported_background_retains_whole_mover_exclusion():
    m = UnknownObstacleMemory(UnknownObstacleConfig(supported_background_prior=True))
    acquire(m)
    snapshot = merged_update(m, FAR_TAIL)
    assert snapshot.regions[0].mode == 'unknown_motion_reachable_surface_prior'
    assert not snapshot.background_motion_proofs


def test_one_other_unconfirmed_mover_still_contradicts_background():
    m = memory()
    acquire(m)
    mixed = np.vstack((BASE, FAR_TAIL))
    close = np.array([[1.18, 0.], [1.19, .01], [1.20, .02]])
    points = np.vstack((mixed, close))
    snapshot = m.update(points, np.r_[np.full(len(mixed), 2), np.full(len(close), 3)],
        (track(2, mixed, (.5, 0.)), track(3, close, (.5, 0.))), {2: GUIDANCE, 3: GUIDANCE},
        source_time=1.4, source_frame=14, robot_xy=(0., 0.))
    assert snapshot.regions[0].mode == 'unknown_motion_reachable_surface_prior'
    assert {p['stationary_overlap_only'] for p in snapshot.background_motion_proofs} == {False, True}


def test_already_qualified_old_geometry_persists_without_new_99_percent_support():
    m = memory()
    acquire(m)
    small_static = BASE[:90]
    points = np.vstack((small_static, FAR_TAIL))
    snapshot = m.update(points, np.full(len(points), 2), (track(2, points, (.5, 0.)),),
        {2: GUIDANCE}, source_time=1.4, source_frame=14, robot_xy=(0., 0.))
    old, = snapshot.regions
    assert old.original.source_time == 1.2
    np.testing.assert_array_equal(old.original.points, PARTIAL)
    assert old.mode != 'unknown_motion_reachable_surface_prior'


@pytest.mark.parametrize('key,value', [('current_source_time', 9.),
    ('repeated_stationary_point_count', 0), ('stationary_overlap_only', False),
    ('minimum_unconfirmed_distance_m', .01), ('support_source_times', [1.3, 1.4])])
def test_provenance_audit_rejects_tampered_stationary_overlap_proof(key, value):
    m = memory()
    acquire(m)
    snapshot = merged_update(m, FAR_TAIL)
    proof = dict(snapshot.background_motion_proofs[0], **{key:value})
    snapshot = replace(snapshot, background_motion_proofs=(proof,))
    assert not summarize_unknown_obstacles([proof_record(snapshot)])['passed']
