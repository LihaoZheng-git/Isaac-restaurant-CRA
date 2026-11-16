from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from robot.perception.unknown_obstacle_memory import (
    UnknownObstacleConfig, UnknownObstacleMemory,
)
from robot.perception.stationary_furniture_memory import StationaryFurnitureMemory
from robot.evaluation.nearby_track_loss_guard import NearbyTrackLossConfig, NearbyTrackLossGuard
from robot.safety.class_safety_policy import ClassSafetyPolicy
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig, build_local_occupancy_grid
from test_stationary_furniture_memory import POINTS, GUIDANCE, track


def observe(m, t, *, key=1, points=POINTS, furniture=None, **kwargs):
    tracks = (track(key, points),) if len(points) else ()
    ids = np.full(len(points), key, dtype=np.int64)
    if furniture is not None:
        furniture.update(points, ids, tracks, {key: GUIDANCE}, source_time=t,
            source_frame=round(t*10), robot_xy=(0., 0.))
    return m.update(points, ids, tracks, {key: GUIDANCE}, source_time=t,
        source_frame=round(t*10), robot_xy=(0., 0.), furniture_memory=furniture, **kwargs)


def acquire(m):
    f = StationaryFurnitureMemory()
    for t in (1., 1.1, 1.2):
        observe(m, t, furniture=f)
    assert m.records[1].stationary_support_time == 1.2
    return f


def raster(snapshot, *, bounds=(-5, 5), anchor=(0., 0.), yaw=0.):
    cfg = LocalOccupancyGridConfig(resolution=.1, minimum_x=bounds[0], maximum_x=bounds[1],
        minimum_y=bounds[0], maximum_y=bounds[1])
    points, ids, parameters = [], [], {}
    for r in snapshot.regions:
        xy, _, _, _ = r.cells(cfg, anchor, yaw)
        points.extend(xy)
        ids.extend([r.geometry_id]*len(xy))
        parameters[r.geometry_id] = r.original.guidance.safety_parameters
    return build_local_occupancy_grid(np.asarray(points).reshape((-1, 2)),
        point_track_ids=np.array(ids, dtype=np.int64),
        class_safety_policy=ClassSafetyPolicy({'unknown': GUIDANCE.safety_parameters}),
        track_safety_parameters=parameters, config=cfg)


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), 1.01, True])
def test_invalid_phi_rejects(value):
    with pytest.raises(ValueError):
        UnknownObstacleConfig(unknown_phi=value)


def test_fixed_phi_and_higher_old_risk_preserved_without_changing_lambda():
    from test_tracking_uncertainty import guidance
    cfg = UnknownObstacleConfig()
    assert cfg.guidance(guidance(.2)).risk == .68359375
    assert cfg.guidance(guidance(.9)).risk == .9
    assert cfg.guidance(guidance(.9)).tangential_strength == 1.25
    assert cfg.guidance(guidance(.9)).boundary_flux == pytest.approx(-1.58)


def test_missing_qualified_surface_becomes_obstacle_without_renewed_source():
    m = UnknownObstacleMemory()
    acquire(m)
    s = observe(m, 1.5, points=np.empty((0, 2)))
    r, = s.regions
    assert r.mode == 'retained_stationary_geometry_prior'
    assert r.radius_m == pytest.approx(.0515)
    assert r.original.source_time == r.original.stationary_support_time == 1.2
    np.testing.assert_array_equal(r.original.points, POINTS)
    assert s.coverage(1, 1.2, POINTS.mean(0), raster(s), (0., 0.), 0.)


def test_stationary_expiry_switches_to_motion_geometry_instead_of_disappearing():
    m = UnknownObstacleMemory()
    acquire(m)
    s = observe(m, 12., points=np.empty((0, 2)))
    r, = s.regions
    assert r.mode == 'unknown_motion_reachable_surface_prior'
    assert r.radius_m == pytest.approx(.1+.8+1.55)
    assert r.original.source_time == 1.2


def test_unqualified_single_observation_uses_motion_prior_not_zero_velocity():
    m = UnknownObstacleMemory()
    observe(m, 1.)
    s = observe(m, 1.2, points=np.empty((0, 2)))
    assert s.regions[0].radius_m == pytest.approx(.1+.2+1.55)
    assert s.regions[0].original.stationary_support_time is None


def test_old_surface_full_geometry_replacement_has_original_epoch_and_current_owner():
    m = UnknownObstacleMemory()
    observe(m, 1.)
    s = observe(m, 1.2, key=2, points=POINTS+[.07, 0.])
    r, = s.regions
    assert r.mode == 'current_whole_surface_geometry_replacement'
    assert r.original.source_time == 1.
    assert r.replacement_track_id == 2
    assert 1 not in m.records
    assert set(m.records) == {2}


@pytest.mark.parametrize('failure', ['partial', 'ambiguous', 'unclustered'])
def test_incomplete_or_ambiguous_geometry_keeps_expanding_obstacle(failure):
    m = UnknownObstacleMemory()
    observe(m, 1.)
    p = POINTS+[.07, 0.]
    if failure == 'partial':
        p = p + [0, .2]
    ids = np.full(len(p), 2)
    tracks = [track(2, p)]
    if failure in ('ambiguous', 'unclustered'):
        p = np.vstack((p, POINTS))
        ids = np.r_[ids, np.full(3, 3 if failure == 'ambiguous' else -1)]
        if failure == 'ambiguous':
            tracks.append(track(3))
    s = m.update(p, ids, tracks, {2: GUIDANCE, 3: GUIDANCE}, source_time=1.2,
        source_frame=12, robot_xy=(0., 0.))
    assert s.regions[0].mode == 'unknown_motion_reachable_surface_prior'
    assert s.regions[0].replacement_track_id is None
    assert 1 in m.records


def test_repeated_positive_support_resolves_old_static_fragment_across_ids():
    m = UnknownObstacleMemory()
    acquire(m)
    # A new cluster initially has too little matching support; old source stays.
    for t in (1.5, 1.7):
        observe(m, t, key=2, points=POINTS+[.11, 0.])
    for t in (1.9, 2.1, 2.3):
        s = observe(m, t, key=2, points=POINTS+[.07, 0.])
        if 1 not in m.records:
            break
    r = next(r for r in s.regions if r.original.track_id == 1)
    assert r.mode == 'current_whole_surface_geometry_replacement'
    assert r.original.source_time == 1.2
    assert r.support_distance_m == pytest.approx(.07)


def test_human_exclusion_invalidates_static_assumption_without_erasing_geometry():
    m = UnknownObstacleMemory()
    acquire(m)
    s = observe(m, 1.4, points=np.empty((0, 2)), human_regions=[(np.array([1., 0.]), .65)])
    assert s.regions[0].mode == 'unknown_motion_reachable_surface_prior'
    assert s.regions[0].original.source_time == 1.2


@pytest.mark.parametrize('failure', ['risk', 'occupancy', 'source', 'center'])
def test_guard_requires_actual_current_field_not_just_unknown_label(failure):
    m = UnknownObstacleMemory()
    acquire(m)
    s = observe(m, 1.5, points=np.empty((0, 2)))
    g = raster(s)
    time, center = 1.2, POINTS.mean(0)
    if failure == 'risk':
        g = replace(g, cell_risks=np.where(g.occupied_mask, .01, 0.))
    elif failure == 'occupancy':
        g = replace(g, occupied_mask=np.zeros(g.config.shape, bool),
            cell_track_ids=np.full(g.config.shape, -1, dtype=np.int64),
            cell_risks=np.zeros(g.config.shape), cell_boundary_fluxes=np.zeros(g.config.shape))
    elif failure == 'source':
        time = 1.1
    else:
        center = center + [.1, 0]
    assert s.coverage(1, time, center, g, (0., 0.), 0.) is None


def test_local_clipping_is_reported_and_work_bounded_without_claiming_free_space():
    m = UnknownObstacleMemory()
    observe(m, 1.)
    s = observe(m, 100., points=np.empty((0, 2)))
    g = raster(s, bounds=(-1, 1))
    assert g.occupied_mask.all()
    proof = s.coverage(1, 1., POINTS.mean(0), g, (0., 0.), 0.)
    assert proof['region_clipped_to_local_field']
    assert proof['checked_cell_count'] == g.occupied_mask.size


def test_track_loss_guard_continues_only_with_valid_unknown_obstacle_field():
    m = UnknownObstacleMemory()
    acquire(m)
    s = observe(m, 1.5, points=np.empty((0, 2)))
    guard = NearbyTrackLossGuard(NearbyTrackLossConfig(allow_unknown_obstacle_planning=True))
    guard.observed = {1: (POINTS.mean(0), 1.2)}
    result = guard.observe((), source_time=1.5, robot_world_xy=(0., 0.),
        current_unknown_obstacle_snapshot=s, current_occupancy_grid=raster(s),
        current_grid_anchor_xy=(0., 0.), current_grid_anchor_yaw=0.)
    assert not result['blocked']
    assert result['unknown_obstacle_covered_missing_tracks'][0]['original_source_time'] == 1.2
    strict = NearbyTrackLossGuard()
    strict.observed = {1: (POINTS.mean(0), 1.2)}
    assert strict.observe((), source_time=1.5, robot_world_xy=(0., 0.))['blocked']


def test_replacement_retires_old_query_only_after_field_coverage():
    m = UnknownObstacleMemory()
    observe(m, 1.)
    s = observe(m, 1.2, key=2, points=POINTS+[.07, 0.])
    guard = NearbyTrackLossGuard(NearbyTrackLossConfig(allow_unknown_obstacle_planning=True))
    guard.observed = {1: (POINTS.mean(0), 1.)}
    # Disable the older centroid association by also watching the new ID.
    guard.observed[2] = (POINTS.mean(0)+[.07, 0], 1.1)
    result = guard.observe((track(2, POINTS+[.07, 0]),), source_time=1.2, robot_world_xy=(0., 0.),
        current_unknown_obstacle_snapshot=s, current_occupancy_grid=raster(s),
        current_grid_anchor_xy=(0., 0.), current_grid_anchor_yaw=0.)
    assert not result['blocked'] and 1 not in guard.observed
    assert guard.observed[2][1] == 1.2


def test_actual_worker_transaction_and_raw_occupancy_preserved():
    from test_object_risk_map import mapped_worker
    from test_live_object_perception import packet
    from robot.evaluation.live_checked_wait import StagedLivePerception
    from robot.safety.rolling_local_field import RollingLocalField
    w = mapped_worker()
    w.stationary_furniture_memory = StationaryFurnitureMemory()
    w.unknown_obstacle_memory = UnknownObstacleMemory()
    w.rolling_field = RollingLocalField(w.grid_config)
    first, second = packet().obstacle_xyz, packet().obstacle_xyz+[0., 1.3, 0.]
    for frame, t in enumerate((1., 1.1, 1.2), 1):
        original = w(replace(packet(frame, t), obstacle_xyz=np.vstack((first, second))))
    saved = w.unknown_obstacle_memory
    staged = StagedLivePerception(w)
    result = staged(replace(packet(4, 1.3), obstacle_xyz=second))
    assert w.unknown_obstacle_memory is saved and saved.last_time == 1.2
    assert result.unknown_obstacle_snapshot.regions
    for r in result.unknown_obstacle_snapshot.regions:
        assert result.unknown_obstacle_snapshot.coverage(r.original.track_id,
            r.original.source_time, r.original.center, result.source_occupancy_grid,
            result.field.anchor_position_world_xy, result.field.anchor_yaw)
    assert result.field.all_obstacle_point_count == len(second)
    assert len(result.field.tracking_result.tracked_obstacles) < len(original.field.tracking_result.tracked_obstacles)
    staged.discard()
    assert w.unknown_obstacle_memory.last_time == 1.2
    result = staged(replace(packet(5, 1.4), obstacle_xyz=second))
    staged.accept(result)
    assert w.unknown_obstacle_memory.last_time == 1.4


@pytest.mark.parametrize('case', ['long-parallel', 'long-parallel-near', 'long-parallel-component'])
def test_new_recipe_real_parser_and_only_explicit_option(case, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    old = json.loads((root/'configs/restaurant_service_demo_v2_short_fragment.json').read_text())['commands'][case]
    command = json.loads((root/'configs/restaurant_service_demo_v2_unknown_obstacles.json').read_text())['commands'][case]
    assert command == [*old, '--unknown-obstacle-planning']
    source = root/command[0]
    monkeypatch.setattr(sys, 'argv', [*command, '--output', str(tmp_path/'native')])
    monkeypatch.setattr(sys, 'path', list(sys.path))
    scope = dict(__file__=str(source), __name__='cli_test')
    exec(compile(source.read_text().split('args.output.mkdir', 1)[0], str(source), 'exec'), scope)
    assert scope['args'].unknown_obstacle_planning
    assert scope['args'].unknown_obstacle_phi == .68359375
    assert not (tmp_path/'native').exists()
