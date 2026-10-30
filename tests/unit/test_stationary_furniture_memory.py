from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot.evaluation.nearby_track_loss_guard import NearbyTrackLossConfig, NearbyTrackLossGuard
from robot.evaluation.restaurant_baselines import fixed_class_comparison_policy
from robot.perception.stationary_furniture_memory import (
    ID_OFFSET, FurnitureMemoryConfig, StationaryFurnitureMemory, person_exclusion_points,
)
from robot.safety.local_occupancy_grid import LocalOccupancyGrid, LocalOccupancyGridConfig
from robot.safety.obstacle_tracking import TrackedObstacle


POINTS = np.array([[1., -.03], [1., 0.], [1., .03]])
GUIDANCE = fixed_class_comparison_policy().parameters_for("unknown")


def track(key=1, points=POINTS, velocity=(0., 0.)):
    center = tuple(points.mean(axis=0))
    return TrackedObstacle(key, center, center, velocity, velocity, len(points), 5, True)


def observe(memory, now, *, key=1, points=POINTS, velocity=(0., 0.), **kwargs):
    return memory.update(points, np.full(len(points), key),
        (track(key, points, velocity),) if len(points) else (),
        {key: GUIDANCE}, source_time=now, source_frame=round(now*10), robot_xy=(0., 0.), **kwargs)


def acquire(memory, *, keys=(1, 1, 1)):
    for t, key in zip((1., 1.1, 1.2), keys):
        result = observe(memory, t, key=key)
    return result


def missing(memory, now=1.4, **kwargs):
    return observe(memory, now, points=np.empty((0, 2)), **kwargs)


def grid_for(snapshot, cfg=None, anchor=(0., 0.), yaw=0.):
    cfg = cfg or LocalOccupancyGridConfig(resolution=.05, minimum_x=-1., maximum_x=3., minimum_y=-2., maximum_y=2.)
    occupied = np.zeros(cfg.shape, bool)
    ids = np.full(cfg.shape, -1, dtype=np.int64)
    risks, flux = np.zeros(cfg.shape), np.zeros(cfg.shape)
    points, keys = snapshot.placements(cfg, anchor, yaw)
    cells = np.floor((points-[cfg.minimum_x, cfg.minimum_y])/cfg.resolution).astype(int)
    if len(cells):
        c, r = cells.T
        occupied[r, c], ids[r, c] = True, keys
        risks[r, c], flux[r, c] = GUIDANCE.risk, GUIDANCE.boundary_flux
    return LocalOccupancyGrid(cfg, occupied, ids, risks, flux, len(points), len(points))


def test_repeated_surface_can_survive_id_changes_without_transferring_identity():
    memory = StationaryFurnitureMemory()
    _, audit = acquire(memory, keys=(9, 8, 1))
    assert audit["enrolled_track_ids"] == [1]
    snapshot, _ = missing(memory)
    r = snapshot.surfaces[0]
    assert r.track_id == 1 and r.source_time == 1.2 and r.support_start_time == 1.
    assert r.source_frame == 12
    proof = snapshot.coverage(1, 1.2, (1., 0.), grid_for(snapshot), (0., 0.), 0.)
    assert proof["missing_seconds"] == pytest.approx(.2)
    assert proof["original_risk"] == GUIDANCE.risk
    assert not snapshot.coverage(9, 1.2, (1., 0.), grid_for(snapshot), (0., 0.), 0.)
    later, _ = missing(memory, 2.)
    assert later.surfaces[0].source_time == 1.2
    assert later.radius(r) > snapshot.radius(r)


@pytest.mark.parametrize("problem", ["one_scan", "two_scans", "new_surface", "gap", "moving", "person"])
def test_no_memory_for_unqualified_or_human_surfaces(problem):
    m = StationaryFurnitureMemory()
    if problem in {"one_scan", "two_scans"}:
        observe(m, 1.)
        if problem == "two_scans":
            observe(m, 1.1)
    else:
        observe(m, 1.)
        observe(m, 1.1)
        kwargs = {"points": POINTS + [.2, 0]} if problem == "new_surface" else {}
        if problem == "moving":
            kwargs["velocity"] = (.5, 0.)
        if problem == "person":
            kwargs["excluded_points"] = [[1., 0.]]
        observe(m, 1.6 if problem == "gap" else 1.2, **kwargs)
    assert not missing(m, 1.8)[0].surfaces


@pytest.mark.parametrize("exclusion", ["person", "motion"])
def test_new_person_or_motion_purges_previous_idle_history_and_geometry(exclusion):
    m = StationaryFurnitureMemory()
    acquire(m)
    kwargs = {"excluded_points": [[1., 0.]]} if exclusion == "person" else {"velocity": (.5, 0.)}
    observe(m, 1.3, **kwargs)
    assert not missing(m, 1.4)[0].surfaces
    # No borrowed qualification after the region is clear again.
    assert not observe(m, 1.5)[1]["enrolled_track_ids"]


@pytest.mark.parametrize("problem", ["expired", "wrong_time", "wrong_center", "clipped", "missing_cell", "low_risk", "low_flux"])
def test_guard_proof_requires_full_actual_field_coverage_and_original_source(problem):
    m = StationaryFurnitureMemory()
    acquire(m)
    snapshot, _ = missing(m, 11.3 if problem == "expired" else 1.4)
    cfg = LocalOccupancyGridConfig(resolution=.05, minimum_x=1., maximum_x=3., minimum_y=-2., maximum_y=2.) if problem == "clipped" else None
    grid = grid_for(snapshot, cfg)
    if grid.occupied_mask.any():
        row, col = np.argwhere(grid.occupied_mask)[0]
        if problem == "missing_cell":
            grid.occupied_mask[row, col] = False
        if problem == "low_risk":
            grid.cell_risks[row, col] = 0
        if problem == "low_flux":
            grid.cell_boundary_fluxes[row, col] = 0
    assert snapshot.coverage(1, 1.1 if problem == "wrong_time" else 1.2,
        (1.1, 0.) if problem == "wrong_center" else (1., 0.), grid, (0., 0.), 0.) is None


def test_watchdog_uses_actual_retained_field_and_restores_stop_on_expiry():
    m = StationaryFurnitureMemory()
    acquire(m)
    guard = NearbyTrackLossGuard(NearbyTrackLossConfig(allow_stationary_furniture_memory=True))
    snapshot, _ = observe(m, 1.3)
    guard.observe((track(),), source_time=1.3, robot_world_xy=(0., 0.),
        current_furniture_snapshot=snapshot, current_occupancy_grid=grid_for(snapshot),
        current_grid_anchor_xy=(0., 0.), current_grid_anchor_yaw=0.)
    for now, blocked in ((1.6, False), (2., False), (11.4, True)):
        snapshot, _ = missing(m, now)
        result = guard.observe((), source_time=now, robot_world_xy=(0., 0.),
            current_furniture_snapshot=snapshot, current_occupancy_grid=grid_for(snapshot),
            current_grid_anchor_xy=(0., 0.), current_grid_anchor_yaw=0.)
        assert result["blocked"] == blocked
        assert guard.observed[1][1] == 1.3
        assert bool(result["furniture_prior_covered_missing_tracks"]) != blocked


def test_current_real_reobservation_can_renew_only_geometry_source():
    m = StationaryFurnitureMemory()
    acquire(m)
    missing(m)
    observe(m, 1.5)
    snapshot, _ = missing(m, 1.6)
    assert snapshot.surfaces[0].source_time == 1.5
    assert snapshot.surfaces[0].guidance is GUIDANCE


def test_raster_covers_full_disks_in_rotated_rolling_grid():
    m = StationaryFurnitureMemory()
    acquire(m)
    snapshot, _ = missing(m)
    grid = grid_for(snapshot, anchor=(.2, -.1), yaw=.7)
    assert snapshot.coverage(1, 1.2, (1., 0.), grid, (.2, -.1), .7)
    assert ID_OFFSET + 1 in grid.cell_track_ids


def test_person_mask_excludes_without_claiming_object_association():
    mask = np.array([[True, False], [False, False]])
    frame = SimpleNamespace(source_frame_id=3, source_timestamp_seconds=1.2,
        detections=[SimpleNamespace(semantic_class="person", confidence=.6, mask=mask)])
    projection = SimpleNamespace(source_indices=np.array([0, 1]), pixel_indices_xy=np.array([[0, 0], [1, 1]]))
    points = person_exclusion_points(np.array([[1., 2., 3.], [4., 5., 6.]]), projection, frame, 3, 1.2,
                                    [{"center_world_xy": [2., 3.]}])
    np.testing.assert_array_equal(points, [[1., 2.], [2., 3.]])
    with pytest.raises(ValueError, match="actual current"):
        person_exclusion_points(POINTS, projection, frame, 4, 1.3, [])


def test_rejected_field_does_not_commit_furniture_memory():
    from test_object_risk_map import mapped_worker
    from test_live_object_perception import packet
    from robot.evaluation.live_checked_wait import StagedLivePerception
    from robot.safety.rolling_local_field import RollingLocalField
    worker = mapped_worker()
    worker.stationary_furniture_memory = StationaryFurnitureMemory()
    worker.rolling_field = RollingLocalField(worker.grid_config)
    worker(packet())
    original = worker.stationary_furniture_memory
    staged = StagedLivePerception(worker)
    result = staged(packet(2, 1.1))
    assert result.furniture_memory_snapshot is not None
    assert original.last_time == 1.
    staged.discard()
    result = staged(packet(3, 1.2))
    staged.accept(result)
    assert worker.stationary_furniture_memory.last_time == 1.2
    assert original.last_time == 1.


def test_live_field_keeps_lost_surface_without_claiming_new_track_or_factor():
    from test_object_risk_map import mapped_worker
    from test_live_object_perception import packet
    worker = mapped_worker()
    worker.stationary_furniture_memory = StationaryFurnitureMemory()
    first = packet().obstacle_xyz
    second = first + [0., 1.3, 0.]
    for frame, now in enumerate((1., 1.1, 1.2), 1):
        p = replace(packet(frame, now), obstacle_xyz=np.vstack((first, second)))
        result = worker(p)
    original_ids = {t.track_id for t in result.field.tracking_result.tracked_obstacles}
    result = worker(replace(packet(4, 1.3), obstacle_xyz=second))
    current_ids = {t.track_id for t in result.field.tracking_result.tracked_obstacles}
    lost = (original_ids - current_ids).pop()
    assert ID_OFFSET + lost in result.source_occupancy_grid.cell_track_ids
    assert lost not in {o.attributes.track_id for o in result.objects}
    record = result.furniture_memory_snapshot.surfaces[0]
    assert record.source_time == 1.2
    assert result.field.sensor_timestamp_seconds == 1.3
    assert result.furniture_memory_snapshot.coverage(lost, 1.2, record.centroid,
        result.source_occupancy_grid, (0., 0.), 0.)
    assert result.field.all_obstacle_point_count == len(second)


def test_independent_human_truth_audit_detects_frozen_person_region():
    from robot.evaluation.furniture_memory_validation import audit_snapshot, summarize_memory
    m = StationaryFurnitureMemory()
    acquire(m)
    snapshot, diag = missing(m)
    truth = [{"time": 1., "actor_xyz": [0., 0., 0.]}, {"time": 2., "actor_xyz": [2.5, 0., 0.]}]
    assert not audit_snapshot(snapshot, truth)["passed"]  # source-aligned actor x=1
    assert not audit_snapshot(snapshot, truth[1:])["passed"]
    truth = [{"time": 1., "actor_xyz": [3., 3., 0.]}, {"time": 2., "actor_xyz": [3., 3., 0.]}]
    audit = audit_snapshot(snapshot, truth)
    assert audit["passed"]
    assert not summarize_memory([{"furniture_memory_human_audit": audit,
                                  "stationary_furniture_memory": diag}])["passed"]


@pytest.mark.parametrize("kwargs", [{"maximum_age_seconds": 11}, {"support_distance_m": 0},
                                    {"minimum_support_seconds": .1}, {"maximum_age_seconds": float("nan")}])
def test_config_bounds_are_explicit(kwargs):
    with pytest.raises(ValueError):
        FurnitureMemoryConfig(**kwargs)


def test_duplicate_and_invalid_sources_reject():
    m = StationaryFurnitureMemory()
    observe(m, 1.)
    with pytest.raises(ValueError, match="increasing"):
        observe(m, 1.)
    with pytest.raises(ValueError):
        replace(FurnitureMemoryConfig(), uncertainty_radius_m=False)
