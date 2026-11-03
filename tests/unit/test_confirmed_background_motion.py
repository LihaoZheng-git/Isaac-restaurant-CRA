from copy import deepcopy

import numpy as np
import pytest

from robot.perception.background_motion_evidence import confirm_background_motion
from robot.evaluation.unknown_obstacle_validation import summarize_unknown_obstacles
from test_persistent_background_prior import memory, proof_record
from test_supported_background_prior import acquire
from test_stationary_furniture_memory import GUIDANCE, track


# Asymmetric 2D surface: a moving visible corner rather than a sliding wall crop.
SHAPE = np.array([[0., 0.], [.16, 0.], [.31, .03], [.34, .22],
    [.18, .35], [0., .26], [-.05, .11], [.11, .16]])


def history(offsets=(0., .08, .16), times=(1., 1.2, 1.4), shape=SHAPE):
    return [(t, {2:shape+[x, 0.]}) for t, x in zip(times, offsets)]


def test_consistent_full_shape_translation_is_confirmed():
    proof = confirm_background_motion(history(), 2)
    assert proof['confirmed']
    assert proof['sampled_source_times'] == [1., 1.2, 1.4]
    np.testing.assert_allclose(proof['fits'][-1]['translation_xy'], [.16, 0.], atol=1e-12)


@pytest.mark.parametrize('sources', [history()[:2], history(offsets=(0., .08, 0.)),
    history(offsets=(0., .08, .28)), history(times=(1., 1.2, 1.6)),
    history(offsets=(0., .02, .04)), history(offsets=(0., 0., 0.)),
    history(shape=np.column_stack((np.zeros(8), np.linspace(0., .5, 8)))),
    history(shape=SHAPE[:5])])
def test_ambiguous_sequences_do_not_revoke_static_assumption(sources):
    assert not confirm_background_motion(sources, 2)['confirmed']


def test_missing_or_changed_id_breaks_sequence_without_identity_transfer():
    sources = history()
    sources[1] = (1.2, {3:SHAPE+[.08, 0.]})
    assert not confirm_background_motion(sources, 2)['confirmed']


def test_changed_shape_and_partial_visibility_are_not_translation_evidence():
    sources = history()
    sources[1][1][2] = sources[1][1][2][2:]
    assert not confirm_background_motion(sources, 2)['confirmed']


def test_registration_handles_shuffled_points_and_small_sensor_noise():
    rng = np.random.default_rng(7)
    sources = history()
    for _, clouds in sources:
        clouds[2] = rng.permutation(clouds[2])+rng.uniform(-.002, .002, SHAPE.shape)
    assert confirm_background_motion(sources, 2)['confirmed']


def feed(m, now, cloud, key=2, **kwargs):
    return m.update(cloud, np.full(len(cloud), key), (track(key, cloud, (.4, 0.)),),
        {key:GUIDANCE}, source_time=now, source_frame=round(now*10), robot_xy=(0., 0.), **kwargs)


def test_confirmed_mover_revokes_old_prior_with_original_motion_clock_and_full_geometry():
    m = memory()
    acquire(m)
    old = deepcopy(m.records[1])
    # No current cloud covers the whole original surface. The final translated
    # corner overlaps its previously qualified unsupported tail at (1.18, 0).
    for now, shift in ((1.4, 0.), (1.6, .08), (1.8, .16)):
        snapshot = feed(m, now, SHAPE+[1.02+shift, 0.])
    region = next(r for r in snapshot.regions if r.original.track_id == 1)
    assert region.mode == 'unknown_motion_reachable_surface_prior'
    assert region.original.source_time == region.original.motion_support_time == old.source_time
    assert region.original.stationary_support_time is None
    assert region.original.guidance == old.guidance
    np.testing.assert_array_equal(region.original.points, old.points)
    proof = next(p for p in snapshot.background_motion_proofs if p['track_id'] == 1)
    assert proof['motion_confirmation']['confirmed'] and proof['motion_contradiction']
    report = summarize_unknown_obstacles([proof_record(snapshot)])
    assert report['passed'] and report['confirmed_motion_proofs']


@pytest.mark.parametrize('kind', ['source_gap', 'missing_id', 'person'])
def test_motion_history_cannot_bridge_gaps_or_human_exclusions(kind):
    m = memory()
    acquire(m)
    feed(m, 1.4, SHAPE+[1.02, 0.])
    feed(m, 1.6, SHAPE+[1.10, 0.], key=3 if kind == 'missing_id' else 2,
        **({'excluded_points': [[1.1, 0.]]} if kind == 'person' else {}))
    snapshot = feed(m, 2. if kind == 'source_gap' else 1.8, SHAPE+[1.18, 0.])
    proof = confirm_background_motion(m.motion_history, 2)
    assert not proof['confirmed']
    if kind != 'person':
        assert next(r for r in snapshot.regions if r.original.track_id == 1).mode != 'unknown_motion_reachable_surface_prior'


@pytest.mark.parametrize('field,value', [('confirmed', True), ('source_times', [1., 1.1, 1.4]),
    ('reason', 'consistent_observed_translation')])
def test_audit_rejects_invented_confirmation(field, value):
    m = memory()
    acquire(m)
    snapshot = feed(m, 1.4, SHAPE+[1.18, 0.])
    record = proof_record(snapshot)
    proof = record['unknown_obstacle_memory']['background_motion_proofs'][0]
    proof['motion_confirmation'][field] = value
    # Source-times forgery must be inconsistent with the current source or
    # exceed the original history bounds to be detectably false in a summary.
    if field == 'source_times':
        proof['motion_confirmation'][field] = [0., 1.4]
    assert not summarize_unknown_obstacles([record])['passed']


def test_raw_arrays_and_unqualified_motion_are_unchanged():
    m = memory()
    cloud = SHAPE+[1., 0.]
    before = cloud.copy()
    feed(m, 1., cloud)
    np.testing.assert_array_equal(cloud, before)
    from test_unknown_obstacle_memory import observe
    snapshot = observe(m, 1.2, points=np.empty((0, 2)))
    assert snapshot.regions[0].mode == 'unknown_motion_reachable_surface_prior'


def test_motion_archive_replays_actual_geometric_confirmation(tmp_path):
    from robot.evaluation.unknown_obstacle_validation import UnknownObstacleGeometryArchive
    m = memory()
    acquire(m)
    for now, shift in ((1.4, 0.), (1.6, .08), (1.8, .16)):
        snapshot = feed(m, now, SHAPE+[1.02+shift, 0.])
    archive = UnknownObstacleGeometryArchive(tmp_path).save(snapshot)
    replay = []
    for ref in archive['motion_evidence']:
        with np.load(tmp_path/ref['path'], allow_pickle=False) as stored:
            replay.append((ref['source_time'], {ref['track_id']:stored['observed_points_world_xy']}))
    assert confirm_background_motion(replay, 2)['confirmed']
    assert len(archive['motion_evidence']) == 3
