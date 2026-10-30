from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot.evaluation.nearby_track_loss_guard import NearbyTrackLossConfig, NearbyTrackLossGuard
from robot.evaluation.restaurant_baselines import fixed_class_comparison_policy
from robot.perception.stationary_furniture_memory import (
    ID_OFFSET, FurnitureMemoryConfig, StationaryFurnitureMemory, person_exclusion_points,
)
from robot.perception.semantic_detection import PixelBoundingBox
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


def grid_for(snapshot, cfg=None, anchor=(0., 0.), yaw=0., current=False):
    cfg = cfg or LocalOccupancyGridConfig(resolution=.05, minimum_x=-1., maximum_x=3., minimum_y=-2., maximum_y=2.)
    occupied = np.zeros(cfg.shape, bool)
    ids = np.full(cfg.shape, -1, dtype=np.int64)
    risks, flux = np.zeros(cfg.shape), np.zeros(cfg.shape)
    points, keys = snapshot.placements(cfg, anchor, yaw)
    if current:
        for surface in snapshot.observed_surfaces:
            c, s = np.cos(yaw), np.sin(yaw)
            local = (surface.points-anchor) @ np.array([[c, -s], [s, c]])
            points = np.vstack((points, local))
            keys = np.r_[keys, np.full(len(local), surface.current_track_id)]
    cells = np.floor((points-[cfg.minimum_x, cfg.minimum_y])/cfg.resolution).astype(int)
    if len(cells):
        c, r = cells.T
        occupied[r, c], ids[r, c] = True, keys
        risks[r, c], flux[r, c] = GUIDANCE.risk, GUIDANCE.boundary_flux
    return LocalOccupancyGrid(cfg, occupied, ids, risks, flux, len(points), len(points))


def continuously_observed_after_expiry():
    m = StationaryFurnitureMemory()
    acquire(m)
    # New cluster identity, but the original region has fresh whole support.
    for now in np.arange(13, 114) / 10:
        snapshot, audit = observe(m, now, key=2, points=POINTS+[.01, 0.])
    assert not snapshot.surfaces  # The original ten-second geometry lease expired.
    assert audit["observed_coverage_count"] == 1
    return m, snapshot


def test_fresh_whole_surface_resolves_continuity_without_extending_memory_lease():
    _, snapshot = continuously_observed_after_expiry()
    assert snapshot.placements(grid_for(snapshot).config, (0., 0.), 0.)[0].shape == (0, 2)
    assert snapshot.coverage(1, 1.2, (1., 0.), grid_for(snapshot), (0., 0.), 0.) is None
    proof = snapshot.coverage(1, 1.2, (1., 0.), grid_for(snapshot, current=True), (0., 0.), 0.)
    assert proof["status"] == "inferred"
    assert proof["original_source_time"] == 1.2
    assert proof["current_source_time"] == 11.3
    assert proof["current_source_frame"] == 113
    assert proof["current_track_id"] == 2
    assert proof["maximum_correspondence_distance_m"] == pytest.approx(.01)
    assert proof["original_risk"] == GUIDANCE.risk


def test_independent_human_audit_includes_current_surface_proofs_after_memory_expiry():
    from robot.evaluation.furniture_memory_validation import audit_snapshot
    _, snapshot = continuously_observed_after_expiry()
    truth = [{"time": 11.3, "actor_xyz": [1., 0., 0.]}]
    assert not audit_snapshot(snapshot, truth)["passed"]
    truth[0]["actor_xyz"] = [3., 0., 0.]
    audit = audit_snapshot(snapshot, truth)
    assert audit["passed"] and audit["retained_surfaces"] == 0
    assert audit["observed_surface_proofs"] == 1


@pytest.mark.parametrize("problem", ["missing", "partial", "person", "moving", "gap", "ambiguous", "unclustered"])
def test_observed_surface_coverage_rejects_loss_ambiguity_or_contradiction_after_expiry(problem):
    m, _ = continuously_observed_after_expiry()
    now = 11.7 if problem == "gap" else 11.4
    points = POINTS+[.01, 0.]
    extra = {}
    if problem == "missing":
        points = np.empty((0, 2))
    elif problem == "partial":
        points = POINTS+[0., .1]
    elif problem == "person":
        extra["excluded_points"] = [[1., 0.]]
    elif problem == "moving":
        points = POINTS+[.15, 0.]
        extra["velocity"] = (.5, 0.)
    if problem in {"ambiguous", "unclustered"}:
        other = points+[.001, 0.]
        ids = np.r_[[2]*3, [3 if problem == "ambiguous" else -1]*3]
        tracks = (track(2, points),) + ((track(3, other),) if problem == "ambiguous" else ())
        snapshot, _ = m.update(np.vstack((points, other)), ids, tracks, {2: GUIDANCE, 3: GUIDANCE},
            source_time=now, source_frame=114, robot_xy=(0., 0.))
    else:
        snapshot, _ = observe(m, now, key=2, points=points, **extra)
    assert not snapshot.observed_surfaces
    assert not m.observed_associations
    assert snapshot.coverage(1, 1.2, (1., 0.), grid_for(snapshot), (0., 0.), 0.) is None
    # Later clean scans cannot resurrect the expired old query.
    for t in (now+.1, now+.2, now+.3):
        snapshot, _ = observe(m, t, key=2)
        assert not any(r.original.track_id == 1 for r in snapshot.observed_surfaces)


@pytest.mark.parametrize("problem", ["wrong_time", "wrong_center", "missing_cell", "low_risk", "low_flux", "clipped"])
def test_fresh_surface_proof_requires_actual_current_field_and_original_identity(problem):
    _, snapshot = continuously_observed_after_expiry()
    grid = grid_for(snapshot, current=True)
    row, col = np.argwhere(grid.occupied_mask)[0]
    if problem == "missing_cell":
        grid.occupied_mask[row, col] = False
    elif problem == "low_risk":
        grid.cell_risks[row, col] = 0.
    elif problem == "low_flux":
        grid.cell_boundary_fluxes[row, col] = 0.
    proof = snapshot.coverage(1, 1.1 if problem == "wrong_time" else 1.2,
        (1.1, 0.) if problem == "wrong_center" else (1., 0.), grid,
        (10., 0.) if problem == "clipped" else (0., 0.), 0.)
    assert proof is None


def test_guard_keeps_original_watch_epoch_with_current_coverage_then_stops_on_loss():
    m = StationaryFurnitureMemory()
    acquire(m)
    guard = NearbyTrackLossGuard(NearbyTrackLossConfig(allow_stationary_furniture_memory=True))
    snapshot, _ = observe(m, 1.3)
    def check(snapshot, tracks):
        return guard.observe(tracks, source_time=snapshot.source_time, robot_world_xy=(0., 0.),
            current_furniture_snapshot=snapshot, current_occupancy_grid=grid_for(snapshot, current=True),
            current_grid_anchor_xy=(0., 0.), current_grid_anchor_yaw=0.)
    check(snapshot, (track(),))
    # The covering parent's center is far, so centroid reassociation is unavailable.
    large = np.vstack((POINTS, POINTS+[4., 0.]))
    for now in np.arange(14, 115)/10:
        snapshot, _ = observe(m, now, key=2, points=large)
        result = check(snapshot, (track(2, large),))
        assert not result["blocked"]
        assert guard.observed[1][1] == 1.3
    assert not snapshot.surfaces
    assert result["furniture_observed_covered_missing_tracks"]
    assert not result["furniture_prior_covered_missing_tracks"]
    snapshot, _ = missing(m, 11.5)
    result = check(snapshot, ())
    assert result["blocked"]
    assert result["blocking_tracks"][0]["original_last_seen_time"] == 1.3


def test_repeated_surface_can_survive_id_changes_without_transferring_identity():
    memory = StationaryFurnitureMemory()
    _, audit = acquire(memory, keys=(9, 8, 1))
    assert audit["enrolled_track_ids"] == [1]
    snapshot, _ = missing(memory)
    r = next(r for r in snapshot.surfaces if r.track_id == 1)
    assert r.track_id == 1 and r.source_time == 1.2 and r.support_start_time == 1.
    assert r.source_frame == 12
    proof = snapshot.coverage(1, 1.2, (1., 0.), grid_for(snapshot), (0., 0.), 0.)
    assert proof["missing_seconds"] == pytest.approx(.2)
    assert proof["original_risk"] == GUIDANCE.risk
    assert not snapshot.coverage(9, 1.2, (1., 0.), grid_for(snapshot), (0., 0.), 0.)
    later, _ = missing(memory, 2.)
    assert next(r for r in later.surfaces if r.track_id == 1).source_time == 1.2
    assert later.radius(r) > snapshot.radius(r)


def test_later_real_scans_complete_support_after_id_change_without_source_renewal():
    m = StationaryFurnitureMemory()
    observe(m, 1.)
    observe(m, 1.1)
    extended = np.vstack((POINTS, [[1., .1]]))
    snapshot, audit = observe(m, 1.2, points=extended)
    assert not snapshot.surfaces and not audit["enrolled_track_ids"]
    snapshot, audit = observe(m, 1.3, key=2, points=extended)
    assert not snapshot.surfaces and not audit["confirmed_missing_track_ids"]
    snapshot, audit = observe(m, 1.4, key=2, points=extended)
    assert audit["confirmed_missing_track_ids"] == [1]
    record = snapshot.surfaces[0]
    assert record.source_time == 1.2 and record.source_frame == 12
    assert record.qualification_source_time == 1.4
    assert record.support_start_time == 1.2
    np.testing.assert_array_equal(record.points, extended)
    proof = snapshot.coverage(1, 1.2, extended.mean(axis=0), grid_for(snapshot), (0., 0.), 0.)
    assert proof["qualification_source_time"] == 1.4
    assert proof["missing_seconds"] == pytest.approx(.2)
    assert next(r for r in missing(m, 1.5)[0].surfaces if r.track_id == 1).source_time == 1.2


@pytest.mark.parametrize("problem", ["expired", "partial", "person", "moving", "no_current_support"])
def test_pending_surface_never_qualifies_without_bounded_fresh_full_support(problem):
    m = StationaryFurnitureMemory()
    observe(m, 1.)
    observe(m, 1.1, key=2)
    now = 1.8 if problem == "expired" else 1.2
    kwargs = {}
    points = POINTS
    if problem == "partial":
        points = np.array([[1., -.08], [1., -.09], [1., -.1]])
    elif problem == "person":
        kwargs["excluded_points"] = [[1., 0.]]
    elif problem == "moving":
        points = POINTS + [.15, 0.]
        kwargs["velocity"] = (.5, 0.)
    elif problem == "no_current_support":
        points = POINTS + [1., 0.]
    snapshot, audit = observe(m, now, key=2, points=points, **kwargs)
    assert 1 not in audit["confirmed_missing_track_ids"]
    assert 1 not in [r.track_id for r in snapshot.surfaces]


def test_pending_confirmation_allows_two_bounded_actual_source_intervals():
    m = StationaryFurnitureMemory()
    observe(m, 1.)
    observe(m, 1.2, key=2)
    snapshot, audit = observe(m, 1.5, key=2)
    assert audit["confirmed_missing_track_ids"] == [1]
    record = snapshot.surfaces[0]
    assert record.source_time == 1. and record.qualification_source_time == 1.5
    # A source gap still breaks the proof, even within the total buffer window.
    m = StationaryFurnitureMemory()
    observe(m, 1.)
    observe(m, 1.1, key=2)
    assert not observe(m, 1.5, key=2)[0].surfaces


def test_broken_source_sequence_discards_unqualified_candidate_permanently():
    m = StationaryFurnitureMemory()
    observe(m, 1.)
    observe(m, 1.1)
    _, audit = observe(m, 1.5, key=2)
    assert audit["qualification_history_reset"]
    assert 1 not in m.pending_surfaces
    for now in (1.6, 1.7):
        snapshot, audit = observe(m, now, key=2)
        assert not audit["qualification_history_reset"]
        assert 1 not in audit["confirmed_missing_track_ids"]
        assert not any(r.track_id == 1 for r in snapshot.surfaces)
    # Three new sources can qualify the current surface with its actual epoch.
    assert audit["enrolled_track_ids"] == [2]
    snapshot, _ = missing(m, 1.8)
    assert len(snapshot.surfaces) == 1
    assert snapshot.surfaces[0].source_time == 1.7


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
            kwargs["points"] = POINTS + [.15, 0.]
        if problem == "person":
            kwargs["excluded_points"] = [[1., 0.]]
        observe(m, 1.6 if problem == "gap" else 1.2, **kwargs)
    assert not missing(m, 1.8)[0].surfaces


@pytest.mark.parametrize("exclusion", ["person", "motion"])
def test_new_person_or_motion_purges_previous_idle_history_and_geometry(exclusion):
    m = StationaryFurnitureMemory()
    acquire(m)
    kwargs = {"excluded_points": [[1., 0.]]} if exclusion == "person" else {
        "velocity": (.5, 0.), "points": POINTS + [.1, 0.]}
    observe(m, 1.3, **kwargs)
    assert not missing(m, 1.4)[0].surfaces
    # No borrowed qualification of the newly moved surface.
    assert not observe(m, 1.5, points=POINTS + [.1, 0.])[1]["enrolled_track_ids"]


def test_changed_visibility_centroid_does_not_override_whole_static_surface_support():
    m = StationaryFurnitureMemory()
    big = np.vstack((POINTS, POINTS + [.5, 0.]))
    observe(m, 1., points=big)
    observe(m, 1.1, points=big)
    _, audit = observe(m, 1.2, velocity=(-.7, .4))
    assert audit["centroid_surface_disagreement_track_ids"] == [1]
    assert audit["enrolled_track_ids"] == [1]
    assert missing(m)[0].surfaces[0].source_time == 1.2


def test_all_points_need_support_even_when_centroid_motion_is_a_visibility_artifact():
    m = StationaryFurnitureMemory()
    acquire(m)
    changed = POINTS.copy()
    changed[-1] += [.1, .1]
    _, audit = observe(m, 1.3, points=changed, velocity=(-.7, .4))
    assert not audit["centroid_surface_disagreement_track_ids"]
    assert not missing(m)[0].surfaces


def test_centroid_enrollment_preserves_original_support_outside_resource_radius():
    m = StationaryFurnitureMemory()
    big = np.vstack((POINTS + [1., 0.], POINTS + [1.6, 0.]))
    for t in (1., 1.1, 1.2):
        _, audit = observe(m, t, points=big)
    assert audit["enrolled_track_ids"] == [1]
    assert missing(m)[0].surfaces[0].points[:, 0].max() > m.config.enrollment_distance_m


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
    mask = np.zeros((8, 8), bool)
    mask[3, 3] = True
    frame = SimpleNamespace(source_frame_id=3, source_timestamp_seconds=1.2,
        detections=[SimpleNamespace(semantic_class="person", confidence=.6, mask=mask,
                                    bounding_box=PixelBoundingBox(2., 2., 5., 5.))])
    projection = SimpleNamespace(source_indices=np.array([0, 1]), pixel_indices_xy=np.array([[3, 3], [4, 4]]))
    points, audit = person_exclusion_points(np.array([[1., 2., 3.], [4., 5., 6.]]), projection, frame, 3, 1.2,
                                    [{"center_world_xy": [2., 3.]}])
    np.testing.assert_array_equal(points, [[1., 2.], [2., 3.]])
    assert audit["detections"][0]["accepted"]
    with pytest.raises(ValueError, match="actual current"):
        person_exclusion_points(POINTS, projection, frame, 4, 1.3, [])


@pytest.mark.parametrize("edge", ["mask", "left_box", "right_box", "top_box", "bottom_box"])
def test_unresolved_person_boundary_does_not_erase_furniture_but_body_still_excludes(edge):
    mask = np.zeros((8, 8), bool)
    mask[3, 3] = True
    box = PixelBoundingBox(2., 2., 5., 5.)
    if edge == "mask":
        mask[0, 3] = True
    elif edge == "left_box":
        box = replace(box, x_min=.2)
    elif edge == "right_box":
        box = replace(box, x_max=7.5)
    elif edge == "top_box":
        box = replace(box, y_min=.2)
    else:
        box = replace(box, y_max=7.5)
    frame = SimpleNamespace(source_frame_id=3, source_timestamp_seconds=1.2,
        detections=[SimpleNamespace(semantic_class="person", confidence=.9, mask=mask, bounding_box=box)])
    projection = SimpleNamespace(source_indices=np.array([0]), pixel_indices_xy=np.array([[3, 3]]))
    points, audit = person_exclusion_points(POINTS, projection, frame, 3, 1.2, [])
    assert points.shape == (0, 2)
    assert audit["detections"][0]["reason"] == "unresolved_image_boundary_person_cue"
    m = StationaryFurnitureMemory()
    acquire(m)
    assert missing(m, excluded_points=points)[0].surfaces
    points, audit = person_exclusion_points(POINTS, projection, frame, 3, 1.2,
        [{"center_world_xy": [1., 0.]}])
    assert audit["current_body_center_count"] == 1
    assert not missing(m, 1.5, excluded_points=points)[0].surfaces


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
