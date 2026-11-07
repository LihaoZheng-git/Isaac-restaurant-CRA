"""Bounded geometry witnesses preserve source, ambiguity and field contracts."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from robot.evaluation.unknown_obstacle_validation import (
    summarize_unknown_obstacles, validate_injective_support,
)
from robot.perception.surface_support import whole_surface_replacement
from robot.perception.unknown_obstacle_memory import UnknownObstacleConfig, UnknownObstacleMemory
from test_stationary_furniture_memory import POINTS, track
from test_unknown_obstacle_memory import observe, raster


OLD = np.column_stack((1 + np.arange(5)*.001, np.zeros(5)))
NEW = np.column_stack((np.array([1.080, 1.085, 1.090, 1.095, 1.099]), np.zeros(5)))


def supported(old=OLD, current=NEW, ids=None, **kwargs):
    if ids is None:
        ids = np.full(len(current), 2, dtype=np.int64)
    return whole_surface_replacement(old, current, ids, **kwargs)


def snapshot(enabled=True, time=1.2):
    memory = UnknownObstacleMemory(UnknownObstacleConfig(distinct_surface_witness_matching=enabled))
    observe(memory, 1., points=OLD)
    result = observe(memory, time, key=2, points=NEW)
    return memory, result


def test_sparse_full_matching_recovers_distinct_witnesses_without_new_points():
    assert supported(allow_injective=False) is None
    owner, distance, proof = supported()
    assert owner == 2 and distance == pytest.approx(.08)
    assert proof['nearest_distinct_current_point_count'] == 1
    assert len(set(proof['current_witness_indices'])) == len(OLD)
    assert proof['maximum_witness_distance_m'] <= .10
    assert validate_injective_support(proof, original_points=OLD,
        current_points=NEW, current_ids=np.full(len(NEW), 2))


def test_legacy_acceptance_has_no_new_proof_and_is_identical_when_enabled():
    old = supported(POINTS, POINTS+[.07, 0.], allow_injective=False)
    new = supported(POINTS, POINTS+[.07, 0.], allow_injective=True)
    assert old == new and new[2] is None


def test_source_witness_indices_use_original_arrays_not_sorted_or_owner_local_indices():
    old = OLD[[3, 1, 4, 0, 2]]
    current = np.vstack(([8., 8.], NEW[::-1], [9., 9.]))
    ids = np.array([9, 2, 2, 2, 2, 2, 9])
    _, _, proof = supported(old, current, ids)
    np.testing.assert_array_equal(old[proof['original_unique_indices']], proof['original_unique_xy'])
    np.testing.assert_array_equal(current[proof['current_witness_indices']], proof['current_witness_xy'])
    assert validate_injective_support(proof, original_points=old, current_points=current, current_ids=ids)


def test_exact_duplicate_originals_are_retained_but_not_counted_as_distinct():
    old = np.vstack((OLD, OLD[:2]))
    _, _, proof = supported(old)
    assert proof['original_point_count'] == 7 and proof['original_distinct_point_count'] == 5
    assert validate_injective_support(proof, original_points=old,
        current_points=NEW, current_ids=np.full(5, 2))
    assert supported(np.repeat(OLD[:2], 4, axis=0)) is None


def test_duplicate_current_coordinates_cannot_fabricate_witnesses():
    assert supported(current=np.repeat(NEW[:2], 8, axis=0)) is None


def test_every_distinct_old_point_needs_its_own_witness_not_merely_three_anywhere():
    # Four distinct close candidates plus an unrelated fifth cannot cover five
    # originals injectively, even though all originals have nearest support.
    current = np.vstack((NEW[:4], [1.8, 0.]))
    assert supported(current=current) is None


@pytest.mark.parametrize('failure', ['out_of_bound', 'second_owner', 'unowned'])
def test_distance_and_owner_ambiguity_gates_remain_strict(failure):
    current, ids = NEW.copy(), np.full(5, 2)
    if failure == 'out_of_bound':
        current += [.021, 0]
    elif failure == 'second_owner':
        current = np.vstack((current, NEW))
        ids = np.r_[ids, np.full(5, 3)]
    else:
        current = np.vstack((current, NEW[:3]))
        ids = np.r_[ids, np.full(3, -1)]
    assert supported(current=current, ids=ids) is None


def test_fallback_resource_bounds_reject_before_calling_sparse_solver(monkeypatch):
    import robot.perception.surface_support as module
    def unexpected(*args, **kwargs):
        raise AssertionError('over-budget graph reached solver')
    monkeypatch.setattr(module, 'maximum_bipartite_matching', unexpected)
    old = np.column_stack((1 + np.arange(257)*1e-6, np.zeros(257)))
    current = np.column_stack((1.08 + np.arange(257)*1e-6, np.zeros(257)))
    assert supported(old, current) is None
    monkeypatch.setattr(module, 'MAXIMUM_CANDIDATE_EDGES', 4)
    assert supported() is None


def test_runtime_opt_in_changes_only_additional_geometry_proof():
    off, old = snapshot(False)
    on, new = snapshot(True)
    assert UnknownObstacleConfig().distinct_surface_witness_matching is False
    assert old.regions[0].mode == 'unknown_motion_reachable_surface_prior' and 1 in off.records
    region = new.regions[0]
    assert region.mode == 'current_whole_surface_geometry_replacement' and 1 not in on.records
    assert region.original.source_time == 1. and region.original.source_frame == 10
    np.testing.assert_array_equal(region.original.points, OLD)
    assert region.original.guidance == old.regions[0].original.guidance
    assert region.replacement_track_id == 2 and region.radius_m == .10
    assert region.injective_support['current_source_time'] == 1.2
    assert region.injective_support['current_source_frame'] == 12
    assert region.injective_support['original_risk'] == region.original.guidance.risk
    assert region.original.dynamic_evidence is None and on.records[2].dynamic_evidence is None
    assert 'injective_support' not in old.diagnostic()['regions'][0]


def test_injective_fallback_never_extends_original_gap():
    memory, result = snapshot(True, time=1.351)
    assert result.regions[0].replacement_track_id is None and 1 in memory.records


@pytest.mark.parametrize('bad', [1, None, 'yes'])
def test_opt_in_is_strictly_boolean(bad):
    with pytest.raises(ValueError):
        UnknownObstacleConfig(distinct_surface_witness_matching=bad)


@pytest.mark.parametrize('failure', ['risk', 'occupancy', 'source'])
def test_injective_replacement_still_requires_original_actual_field_coverage(failure):
    _, result = snapshot()
    region = result.regions[0]
    grid = raster(result)
    time = 1.
    if failure == 'risk':
        grid = replace(grid, cell_risks=np.zeros(grid.config.shape))
    elif failure == 'occupancy':
        grid = replace(grid, occupied_mask=np.zeros(grid.config.shape, bool))
    else:
        time = 1.1
    assert result.coverage(1, time, region.original.center, grid, (0., 0.), 0.) is None


def test_original_higher_risk_floor_survives_replacement():
    from test_tracking_uncertainty import guidance
    memory = UnknownObstacleMemory(UnknownObstacleConfig(distinct_surface_witness_matching=True))
    memory.update(OLD, np.full(5, 1), (track(1, OLD),), {1: guidance(.94)},
        source_time=1., source_frame=10, robot_xy=(0., 0.))
    result = memory.update(NEW, np.full(5, 2), (track(2, NEW),), {2: guidance(.2)},
        source_time=1.2, source_frame=12, robot_xy=(0., 0.))
    assert result.regions[0].original.guidance.risk == .94
    assert memory.records[2].guidance.risk == .94


@pytest.mark.parametrize('fault', ['duplicate_witness', 'duplicate_old', 'distance', 'incomplete',
    'index', 'owner', 'coordinate', 'count', 'edge_count', 'nearest_distance', 'bounds'])
def test_independent_archive_certificate_check_rejects_corruption(fault):
    _, _, proof = supported()
    proof = deepcopy(proof)
    ids = np.full(5, 2)
    if fault == 'duplicate_witness':
        proof['current_witness_xy'][1] = proof['current_witness_xy'][0]
    elif fault == 'duplicate_old':
        proof['original_unique_xy'][1] = proof['original_unique_xy'][0]
    elif fault == 'distance':
        proof['current_witness_xy'][0][0] += .2
    elif fault == 'incomplete':
        proof['current_witness_indices'].pop()
    elif fault == 'index':
        proof['current_witness_indices'][0] = 9000
    elif fault == 'owner':
        ids[proof['current_witness_indices'][0]] = 3
    elif fault == 'coordinate':
        proof['current_witness_xy'][0][1] += .001
    elif fault == 'count':
        proof['original_point_count'] += 1
    elif fault == 'edge_count':
        proof['candidate_edge_count'] += 1
    elif fault == 'nearest_distance':
        proof['maximum_nearest_distance_m'] += .001
    else:
        proof['maximum_distance_m'] = .11
    assert not validate_injective_support(proof, original_points=OLD,
        current_points=NEW, current_ids=ids)


def test_summary_requires_exact_new_proof_clocks_and_opt_in():
    _, result = snapshot()
    region = result.regions[0]
    check = result.coverage(1, 1., region.original.center, raster(result), (0., 0.), 0.)
    record = dict(sensor_frame_id=12, diagnostics=dict(unknown_obstacle_memory=result.diagnostic(),
        unknown_obstacle_field_checks=[check]))
    assert summarize_unknown_obstacles([record])['passed']
    assert summarize_unknown_obstacles([record])['injective_surface_proofs'] == 1
    for fault in ('source', 'clock', 'risk', 'opt_in', 'field_proof',
                  'none_field_check', 'empty_field_check'):
        bad = deepcopy(record)
        d = bad['diagnostics']
        p = d['unknown_obstacle_memory']['regions'][0]['injective_support']
        if fault == 'source':
            p['original_source_frame'] += 1
        elif fault == 'clock':
            p['current_source_time'] += .1
        elif fault == 'risk':
            p['original_risk'] = .01
        elif fault == 'opt_in':
            d['unknown_obstacle_memory']['config']['distinct_surface_witness_matching'] = False
        elif fault == 'none_field_check':
            d['unknown_obstacle_field_checks'][0] = None
        elif fault == 'empty_field_check':
            d['unknown_obstacle_field_checks'][0] = {}
        else:
            d['unknown_obstacle_field_checks'][0].pop('injective_support')
        assert not summarize_unknown_obstacles([bad])['passed']
