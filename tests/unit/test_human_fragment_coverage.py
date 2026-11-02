from dataclasses import replace

import numpy as np
import pytest

from robot.evaluation.nearby_track_loss_guard import NearbyTrackLossGuard
from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE, envelope_grid_points
from robot.perception.human_fragment_coverage import (
    HumanFragmentCoverage, HumanFragmentCoverageConfig, apply_coverage_risk_floors,
    validate_coverage_grid,
)
from robot.safety.class_safety_policy import ClassSafetyParameters, ClassSafetyPolicy
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig, build_local_occupancy_grid
from robot.safety.obstacle_tracking import TrackedObstacle
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidanceParameters


CONFIG = HumanFragmentCoverageConfig("separated adult; configured point-motion scope")
GRID = LocalOccupancyGridConfig(resolution=.05, minimum_x=0, maximum_x=3, minimum_y=-1, maximum_y=1)


def data(time=1., fragment=True, *, fragment_points=None, speed=.1):
    parent_points = np.array([[1.+i*.005, .3+j*.005] for i in range(4) for j in range(3)])
    fragment_points = np.array([[1.08, -.02], [1.12, -.02], [1.12, .02], [1.08, .02]]) if fragment_points is None else np.array(fragment_points)
    parent = TrackedObstacle(7, (1., 0.), (1., 0.), (.1, 0.), (.1, 0.), len(parent_points), 10, True)
    tracks = [parent]
    points, ids = parent_points, np.full(len(parent_points), 7)
    if fragment:
        center = tuple(fragment_points.mean(axis=0))
        tracks.append(replace(parent, track_id=8, point_count=len(fragment_points),
            centroid_world_xy=center, velocity_world_xy=(speed, 0.)))
        points, ids = np.vstack((points, fragment_points)), np.r_[ids, np.full(len(fragment_points), 8)]
    body = dict(track_id=7, envelope_track_id=3_000_000_007,
        center_world_xy=[1., 0.], radius_m=.6, source=HUMAN_ENVELOPE_SOURCE,
        status="prior", lidar_timestamp_seconds=time, original_person_camera_timestamp=1.,
        current_support_count=len(parent_points), velocity_world_xy=[.1, 0.])
    return tracks, points, ids, {7:.6, 8:.9}, [body]


def warmed(**kwargs):
    memory = HumanFragmentCoverage(CONFIG)
    for t in (1., 1.1, 1.2, 1.3):
        memory.update(*data(t, **kwargs), source_time=t)
    return memory


def missing(memory, time=1.4):
    return memory.update(*data(time, False), source_time=time)


def grid(body, risk=.9):
    points, ids = envelope_grid_points([body], grid_config=GRID, anchor_xy=np.zeros(2), anchor_yaw=0.)
    params = ClassSafetyParameters(risk=risk, boundary_flux_magnitude=.5+1.2*risk)
    return build_local_occupancy_grid(points, point_track_ids=ids,
        class_safety_policy=ClassSafetyPolicy({"unknown": params}), config=GRID,
        track_safety_parameters={body["envelope_track_id"]: params})


def test_full_region_motion_and_original_risk_proof_never_renews_source():
    memory = warmed()
    a = missing(memory)["covered"][0]
    b = missing(memory, 1.5)["covered"][0]
    assert a["status"] == "prior" and a["original_risk"] == .9
    assert a["original_source_time"] == b["original_source_time"] == 1.3
    assert b["motion_expansion_m"] == pytest.approx(.2)
    assert b["original_person_camera_timestamp"] == 1.
    assert len(memory.positive) == 2 and memory.positive[8]["source_time"] == 1.3
    missing(memory, 1.6)
    assert not missing(memory, 1.7)["covered"]


def test_centroid_inside_is_insufficient_for_expanded_full_extent():
    points = [[1.1, -.01], [1.55, -.01], [1.55, .01], [1.1, .01]]
    memory = warmed(fragment_points=points)
    assert not missing(memory)["covered"]
    assert missing(memory, 1.5)["rejected"][0]["reason"] == "full_region_plus_motion_not_covered"


@pytest.mark.parametrize("problem", ["speed", "warmup", "parent_gap", "parent_missing", "parent_jump", "competitor"])
def test_ambiguity_and_motion_scope_fail_closed(problem):
    memory = warmed(speed=1.1) if problem == "speed" else warmed()
    if problem == "warmup":
        memory = HumanFragmentCoverage(CONFIG)
        memory.update(*data(1.3), source_time=1.3)
    now = 1.5 if problem == "parent_gap" else 1.4
    tracks, points, ids, risks, bodies = data(now, False)
    if problem == "parent_missing":
        bodies = []
    if problem == "parent_jump":
        tracks = [replace(tracks[0], velocity_world_xy=(.5, 0.))]
    if problem == "competitor":
        tracks += [replace(tracks[0], track_id=9, centroid_world_xy=(1.6, 0.))]
        points = np.vstack((points, points + [.6, 0]))
        ids = np.r_[ids, np.full(12, 9)]
        risks[9] = .6
        bodies += [{**bodies[0], "track_id": 9, "envelope_track_id": 3_000_000_009,
                    "center_world_xy": [1.6, 0.]}]
    assert not memory.update(tracks, points, ids, risks, bodies, source_time=now)["covered"]


def test_risk_floor_changes_only_applied_envelope_mapping_and_retains_fixed_lambda():
    audit = missing(warmed())
    key = audit["covered"][0]["envelope_track_id"]
    original = SafeSAGEGuidanceParameters(ClassSafetyParameters(.6, 1.22), 1.25, "right")
    guidance = apply_coverage_risk_floors({key:original}, audit)
    assert guidance[key].risk == .9 and original.risk == .6
    assert guidance[key].boundary_flux == pytest.approx(-1.58)
    assert guidance[key].tangential_strength == 1.25
    assert audit["covered"][0]["original_source_time"] == 1.3


@pytest.mark.parametrize("problem", [None, "lower_risk", "hole", "clipped"])
def test_entire_expanded_region_must_be_occupied_at_previous_risk(problem):
    proof = missing(warmed())["covered"][0]
    occupancy = grid(data(1.4, False)[-1][0], .8 if problem == "lower_risk" else .9)
    if problem == "hole":
        occupancy = replace(occupancy, occupied_mask=np.zeros_like(occupancy.occupied_mask))
    if problem == "clipped":
        proof = {**proof, "original_bounds_min_world_xy": [-.1, 0.]}
    result = validate_coverage_grid(proof, occupancy, np.zeros(2), 0.)
    assert result["covered"] == (problem is None)


def test_prior_continuation_keeps_missing_guard_entry_and_stops_after_original_expiry():
    guard = NearbyTrackLossGuard(fragment_coverage_config=CONFIG)
    def observe(time, fragment):
        tracks, points, ids, risks, bodies = data(time, fragment)
        return guard.observe(tracks, source_time=time, robot_world_xy=(.4, 0.),
            current_points_world_xy=points, current_point_track_ids=ids,
            current_points_source_time=time, current_track_risks=risks,
            current_body_envelopes=bodies, current_occupancy_grid=grid(bodies[0]),
            current_grid_anchor_xy=np.zeros(2), current_grid_anchor_yaw=0.)
    for t in (1., 1.1, 1.2, 1.3):
        observe(t, True)
    observe(1.4, False)
    result = observe(1.5, False)
    assert not result["blocked"] and result["prior_covered_missing_tracks"]
    assert not result["observed_fragment_merges"]
    assert guard.observed[8][1] == 1.3
    observe(1.6, False)
    assert observe(1.7, False)["blocked"]


@pytest.mark.parametrize("changes", [{"scope":""}, {"maximum_missing_seconds":.4},
    {"maximum_point_speed_mps":0}, {"minimum_observations":3}, {"maximum_observation_gap_seconds":.2}])
def test_invalid_prior_scope_rejected(changes):
    with pytest.raises(ValueError):
        replace(CONFIG, **changes)


def test_real_field_transaction_keeps_coverage_memory_uncommitted_on_failed_solve(monkeypatch):
    from test_object_risk_map import mapped_worker, multi_packet
    import robot.evaluation.multi_object_perception as module
    runtime = mapped_worker()
    runtime.human_fragment_coverage = HumanFragmentCoverage(CONFIG)
    first = runtime(multi_packet())
    original = runtime.human_fragment_coverage
    assert original.last_time == 1.
    assert len(original.positive) == 2
    assert first.diagnostics["human_fragment_coverage_prior"]["covered"] == []
    def fail(*args, **kwargs):
        raise RuntimeError("injected field solve failure")
    monkeypatch.setattr(module, "build_safe_sage_field_pipeline", fail)
    with pytest.raises(RuntimeError, match="injected"):
        runtime(multi_packet(2, 1.1))
    assert runtime.human_fragment_coverage is original
    assert original.last_time == runtime.previous.sensor_timestamp_seconds == 1.


def test_camera_source_epoch_is_separate_from_lidar_and_future_identity_rejected():
    memory = HumanFragmentCoverage(CONFIG)
    tracks, points, ids, risks, bodies = data()
    bodies[0]["original_person_camera_timestamp"] = 1.08
    memory.update(tracks, points, ids, risks, bodies, source_time=1., camera_timestamp_seconds=1.08)
    assert memory.previous_bodies[7]["original_person_camera_timestamp"] == 1.08
    tracks, points, ids, risks, bodies = data(1.1)
    bodies[0]["original_person_camera_timestamp"] = 1.3
    with pytest.raises(ValueError, match="validated human"):
        memory.update(tracks, points, ids, risks, bodies, source_time=1.1, camera_timestamp_seconds=1.18)


def test_rotated_grid_and_degenerate_region_do_not_hide_unoccupied_cells():
    proof = missing(warmed())["covered"][0]
    body = data(1.4, False)[-1][0]
    config = replace(GRID, minimum_x=-2., maximum_x=2., minimum_y=-2., maximum_y=2.)
    points, ids = envelope_grid_points([body], grid_config=config, anchor_xy=np.zeros(2), anchor_yaw=.4)
    params = ClassSafetyParameters(.9, 1.58)
    occupancy = build_local_occupancy_grid(points, point_track_ids=ids, config=config,
        class_safety_policy=ClassSafetyPolicy({"unknown": params}))
    assert validate_coverage_grid(proof, occupancy, np.zeros(2), .4)["covered"]
    point_proof = {**proof, "original_bounds_min_world_xy": [1.1, 0.], "original_bounds_max_world_xy": [1.1, 0.]}
    assert validate_coverage_grid(point_proof, occupancy, np.zeros(2), .4)["covered"]


SHORT = replace(CONFIG, short_fragment_coverage=True)


def new_fragment(time=1., **kwargs):
    tracks, points, ids, risks, bodies = data(time, **kwargs)
    tracks[-1] = replace(tracks[-1], observation_count=1, matched_previous_track=False,
                         velocity_world_xy=(0., 0.))
    return tracks, points, ids, risks, bodies


def test_short_mode_covers_new_fragment_across_point_two_seconds_not_observed_merge():
    for config, expected in ((CONFIG, False), (SHORT, True)):
        memory = HumanFragmentCoverage(config)
        memory.update(*new_fragment(), source_time=1.)
        result = missing(memory, 1.2)
        assert bool(result['covered']) is expected
        if expected:
            proof = result['covered'][0]
            assert proof['original_observation_count'] == 1
            assert proof['original_source_time'] == proof['original_parent_source_time'] == 1.
            assert proof['original_parent_camera_timestamp'] == 1.
            assert proof['motion_expansion_m'] == pytest.approx(.2)
            assert proof['status'] == 'prior'
            assert memory.positive[8]['source_time'] == 1.
            assert missing(memory, 1.3)['covered'][0]['original_source_time'] == 1.
            assert not missing(memory, 1.4)['covered']


@pytest.mark.parametrize('problem', ['original_extent', 'original_parent_missing',
    'current_extent', 'current_parent_missing', 'parent_gap', 'parent_motion', 'measured_fragment_motion'])
def test_short_fragment_still_requires_both_whole_regions_and_bounded_parent(problem):
    memory = HumanFragmentCoverage(SHORT)
    initial = list(new_fragment(fragment_points=[[.5, 0.], [1.5, 0.]]) if problem == 'original_extent' else new_fragment())
    if problem == 'original_extent':
        initial[-1][0]['radius_m'] = .4
    if problem == 'original_parent_missing':
        initial[-1] = []
    if problem == 'measured_fragment_motion':
        initial[0][-1] = replace(initial[0][-1], observation_count=2, velocity_world_xy=(1.1, 0.))
    memory.update(*initial, source_time=1.)
    now = 1.36 if problem == 'parent_gap' else 1.2
    current = list(data(now, False))
    if problem == 'current_parent_missing':
        current[-1] = []
    if problem == 'current_extent':
        current[-1][0]['center_world_xy'] = [.5, 0.]
    if problem == 'parent_motion':
        current[0][0] = replace(current[0][0], velocity_world_xy=(.5, 0.))
    assert not memory.update(*current, source_time=now)['covered']


def add_competitor(values):
    tracks, points, ids, risks, bodies = values
    tracks = [*tracks, replace(tracks[0], track_id=9, centroid_world_xy=(1.7, 0.))]
    points, ids = np.vstack((points, points[:12]+[.7, 0.])), np.r_[ids, np.full(12, 9)]
    risks = {**risks, 9:.6}
    bodies = [*bodies, {**bodies[0], 'track_id':9, 'envelope_track_id':3_000_000_009,
                       'center_world_xy':[1.7, 0.]}]
    return tracks, points, ids, risks, bodies


@pytest.mark.parametrize('at_original', [False, True])
def test_short_fragment_partial_competing_envelope_at_either_epoch_rejects(at_original):
    memory = HumanFragmentCoverage(SHORT)
    values = new_fragment()
    memory.update(*(add_competitor(values) if at_original else values), source_time=1.)
    values = data(1.2, False)
    assert not memory.update(*(values if at_original else add_competitor(values)), source_time=1.2)['covered']


def test_short_fragment_failed_parent_continuity_cannot_revive_old_record():
    memory = HumanFragmentCoverage(SHORT)
    memory.update(*new_fragment(), source_time=1.)
    values = list(data(1.1, False))
    values[-1] = []
    assert not memory.update(*values, source_time=1.1)['covered']
    assert not missing(memory, 1.2)['covered']
    assert not missing(memory, 1.3)['covered']
    assert memory.positive[8]['source_time'] == 1.


@pytest.mark.parametrize('bad_grid', [False, True])
def test_short_guard_requires_current_field_and_keeps_original_missing_watch(bad_grid):
    guard = NearbyTrackLossGuard(fragment_coverage_config=SHORT)
    def observe(t, first):
        tracks, points, ids, risks, bodies = new_fragment(t) if first else data(t, False)
        occupancy = grid(bodies[0], .8 if bad_grid else .9)
        return guard.observe(tracks, source_time=t, robot_world_xy=(.4, 0.),
            current_points_world_xy=points, current_point_track_ids=ids,
            current_points_source_time=t, current_track_risks=risks,
            current_body_envelopes=bodies, current_occupancy_grid=occupancy,
            current_grid_anchor_xy=np.zeros(2), current_grid_anchor_yaw=0.)
    observe(1., True)
    result = observe(1.2, False)
    assert result['blocked'] is bad_grid
    assert not result['observed_fragment_merges']
    assert guard.observed[8][1] == 1.
    if not bad_grid:
        assert result['prior_covered_missing_tracks']
        observe(1.3, False)
        assert observe(1.4, False)['blocked']


def test_short_provenance_audit_rejects_clock_renewal_or_missing_grid_check():
    from copy import deepcopy
    from robot.evaluation.human_fragment_validation import summarize_short_fragment_coverage
    memory = HumanFragmentCoverage(SHORT)
    memory.update(*new_fragment(), source_time=1.)
    audit = missing(memory, 1.2)
    proof = audit['covered'][0]
    proof['field_coverage'] = validate_coverage_grid(proof, grid(data(1.2, False)[-1][0]), np.zeros(2), 0.)
    assert summarize_short_fragment_coverage([{'human_fragment_coverage_prior':audit}])['passed']
    for key in ('original_parent_source_time', 'current_source_time'):
        broken = deepcopy(audit)
        broken['covered'][0][key] += .1
        assert not summarize_short_fragment_coverage([{'human_fragment_coverage_prior':broken}])['passed']
    del proof['field_coverage']
    assert not summarize_short_fragment_coverage([{'human_fragment_coverage_prior':audit}])['passed']
