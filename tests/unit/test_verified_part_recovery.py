from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest

from robot.perception.human_body_envelope import HumanBodyEnvelope, HumanBodyEnvelopeConfig
from robot.perception.human_track_recovery import HumanTrackRecoveryConfig
from robot.perception.unknown_obstacle_memory import UnknownObstacleConfig, UnknownObstacleMemory
from robot.perception.verified_dynamic_evidence import verified_human_associations
from robot.perception.verified_part_recovery import apply_recovery_risk_floors
from robot.evaluation.unknown_obstacle_validation import summarize_unknown_obstacles
from robot.safety.class_safety_policy import ClassSafetyParameters
from test_visual_body_fragments import fixture, associate
from test_persistent_background_prior import proof_record


def setup(enabled=True):
    args = list(fixture())
    tracking, points = args[:2]
    m = HumanBodyEnvelope(HumanBodyEnvelopeConfig(centroid_uncertainty_m=.25),
        recovery_config=HumanTrackRecoveryConfig(verified_part_recovery=enabled), planning_uncertainty=True)
    bodies = m.update(tracking, {1:dict(camera_timestamp_seconds=1.05, camera_frame_id=4)},
        camera_timestamp_seconds=1.05, positive_surfaces={})
    args[4] = bodies
    parts = associate(args)
    assert len(parts) == 1
    m.remember_visual_parts(parts, 1.)
    cfg = UnknownObstacleConfig(supported_background_prior=True, persistent_background_prior=True,
        unclassified_static_prior=True, require_verified_dynamic_evidence=True)
    um = UnknownObstacleMemory(cfg)
    um.update(points[:, :2], tracking.point_track_ids, tracking.tracked_obstacles,
        {t.track_id:cfg.guidance() for t in tracking.tracked_obstacles}, source_time=1., source_frame=10,
        robot_xy=(0., 0.), verified_human_associations=verified_human_associations(tracking, bodies, parts))
    return m, um, tracking, points


def next_scan(initial, points, now=1.2, key=2, shift=.16):
    xy = points[:40, :2]+[shift, 0.]
    t = replace(initial.tracked_obstacles[0], track_id=key, centroid_world_xy=tuple(xy.mean(0)),
        centroid_robot_xy=tuple(xy.mean(0)), observation_count=5, velocity_world_xy=(.8, 0.))
    tracking = SimpleNamespace(timestamp_seconds=now, point_track_ids=np.full(40, key), tracked_obstacles=(t,))
    fresh = {key:dict(camera_timestamp_seconds=now+.05, camera_frame_id=round(now*10))}
    return tracking, xy, fresh


def update(m, um, scan):
    tracking, xy, fresh = scan
    now = tracking.timestamp_seconds
    bodies = m.update(tracking, fresh, camera_timestamp_seconds=now+.05,
        positive_surfaces=um.records, robot_world_xy=(0., 0.))
    recoveries = [p for p in m.recovery_audit if p['decision']=='reacquired_verified_visual_part_geometry']
    guidance = {t.track_id:um.config.guidance() for t in tracking.tracked_obstacles}
    guidance.update({b['envelope_track_id']:um.config.guidance() for b in bodies})
    audit = [dict(track_or_group_id=k, cra_phi=.1) for k in guidance]
    apply_recovery_risk_floors(guidance, audit, recoveries, bodies)
    s = um.update(xy, tracking.point_track_ids, tracking.tracked_obstacles,
        {t.track_id:guidance[t.track_id] for t in tracking.tracked_obstacles}, source_time=now,
        source_frame=round(now*10), robot_xy=(0., 0.),
        verified_human_associations=verified_human_associations(tracking, bodies, ()),
        verified_body_recoveries=recoveries, current_verified_bodies=bodies)
    return s, bodies, guidance, audit


def test_verified_swap_and_return_keep_one_identity_and_all_old_geometry():
    m, um, initial, points = setup()
    for now, key, shift in ((1.2, 2, .16), (1.4, 1, .32)):
        old = um.records[1 if key==2 else 2]
        s, bodies, _, _ = update(m, um, next_scan(initial, points, now, key, shift))
        assert set(m.identities) == {key} and not m.uncertainty_regions
        r = next(r for r in s.regions if r.verified_recovery)
        assert r.mode == 'current_verified_part_geometry_replacement'
        assert r.original.source_time == old.source_time and r.original.source_frame == old.source_frame
        assert r.radius_m == pytest.approx(.2)
        np.testing.assert_array_equal(r.original.points, old.points)
        record = proof_record(s)
        record.update(human_track_recovery_config=asdict(m.recovery_config),
            human_body_envelope_priors=bodies, human_track_recovery=m.recovery_audit)
        assert summarize_unknown_obstacles([record])['passed']
        assert bodies[0]['original_person_camera_timestamp'] == now+.05  # actual fresh accepted cue
        assert r.verified_recovery['visual_part_proof']['source_timestamp_seconds'] == 1.


def test_default_policy_preserves_fresh_cue_exclusion_and_uncertainty():
    m, um, initial, points = setup(False)
    update(m, um, next_scan(initial, points))
    assert set(m.identities) == {1, 2}
    assert [r.track_id for r in m.uncertainty_regions] == [1]


@pytest.mark.parametrize('problem', ['no_link', 'old_link', 'long_gap', 'incomplete_geometry',
    'wrong_surface_epoch', 'jump', 'fast', 'expired_identity', 'competing_old_identity', 'competing_current_track'])
def test_missing_or_ambiguous_evidence_cannot_retire_old_human(problem):
    m, um, initial, points = setup()
    t, xy, fresh = next_scan(initial, points)
    if problem == 'no_link':
        m.visual_part_links = ()
    elif problem == 'old_link':
        m.visual_part_links = ({**m.visual_part_links[0], 'source_timestamp_seconds': .5},)
    elif problem == 'long_gap':
        t, xy, fresh = next_scan(initial, points, now=1.36)
    elif problem == 'incomplete_geometry':
        um.records[1] = replace(um.records[1], points=np.vstack((um.records[1].points, [2., 0.])))
    elif problem == 'wrong_surface_epoch':
        um.records[1] = replace(um.records[1], source_time=.9)
    elif problem == 'expired_identity':
        m.config = replace(m.config, maximum_identity_age_seconds=.1)
    elif problem == 'competing_old_identity':
        m.identities[9] = deepcopy(m.identities[1])
    elif problem == 'competing_current_track':
        t.tracked_obstacles += (replace(t.tracked_obstacles[0], track_id=9),)
        t.point_track_ids = np.r_[t.point_track_ids, np.full(40, 9)]
        xy = np.vstack((xy, xy))
        um.records.pop(2)  # avoid unrelated coverage in this identity-only check
    else:
        delta = dict(centroid_world_xy=(2., 0.)) if problem=='jump' else dict(velocity_world_xy=(1.1, 0.))
        t.tracked_obstacles = (replace(t.tracked_obstacles[0], **delta),)
    if problem == 'competing_current_track':
        original = deepcopy(m.identities)
        with pytest.raises(ValueError, match='ambiguous competing'):
            m.update(t, fresh, camera_timestamp_seconds=t.timestamp_seconds+.05,
                positive_surfaces=um.records, robot_world_xy=(0., 0.))
        assert set(m.identities) == set(original)
        return
    m.update(t, fresh, camera_timestamp_seconds=t.timestamp_seconds+.05,
        positive_surfaces=um.records, robot_world_xy=(0., 0.))
    assert 1 in m.identities
    assert not any(p['decision']=='reacquired_verified_visual_part_geometry' for p in m.recovery_audit)
    assert any(r.track_id==1 for r in m.uncertainty_regions)


def test_recovered_current_body_preserves_higher_risk_flux_and_original_cra():
    m, um, initial, points = setup()
    old = um.records[1]
    um.records[1] = replace(old, guidance=replace(old.guidance, safety_parameters=ClassSafetyParameters(.95, 2.)))
    s, bodies, guidance, audit = update(m, um, next_scan(initial, points))
    for key in (2, 3_000_000_002):
        assert guidance[key].risk == .95 and guidance[key].boundary_flux_magnitude == 2.
        assert guidance[key].tangential_strength == 1.25
    assert bodies[0]['applied_cra_phi'] == .95
    assert all(row['cra_phi']==.1 for row in audit)
    assert next(r for r in s.regions if r.verified_recovery).original.guidance.risk == .95


@pytest.mark.parametrize('tamper', ['extent', 'source', 'body', 'origin'])
def test_report_audit_rejects_broken_recovery_provenance(tamper):
    m, um, initial, points = setup()
    s, bodies, _, _ = update(m, um, next_scan(initial, points))
    record = deepcopy(proof_record(s))
    record.update(human_track_recovery_config=asdict(m.recovery_config),
        human_body_envelope_priors=deepcopy(bodies), human_track_recovery=deepcopy(m.recovery_audit))
    r = next(r for r in record['unknown_obstacle_memory']['regions'] if r['verified_recovery'])
    if tamper == 'extent':
        r['verified_recovery']['geometry_coverage']['whole_bounds_plus_motion_extent_m'] = 0.
    elif tamper == 'source':
        r['verified_recovery']['geometry_coverage']['original_source_time'] = .9
    elif tamper == 'body':
        record['human_body_envelope_priors'][0]['radius_m'] = 1.
    else:
        record['human_track_recovery'] = []
    assert not summarize_unknown_obstacles([record])['passed']


@pytest.mark.parametrize('case', ['long-parallel', 'long-parallel-near', 'long-parallel-component'])
def test_real_native_parser_and_explicit_dependencies(case, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    old = json.loads((root/'configs/restaurant_service_demo_v2_verified_dynamic.json').read_text())['commands'][case]
    cmd = json.loads((root/'configs/restaurant_service_demo_v2_part_recovery.json').read_text())['commands'][case]
    assert cmd == [*old, '--human-verified-part-recovery']
    source = root/cmd[0]
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(sys, 'argv', [*cmd, '--output', str(tmp_path/'native')])
    code = compile(source.read_text().split('args.output.mkdir', 1)[0], str(source), 'exec')
    scope = dict(__file__=str(source), __name__='cli_test')
    exec(code, scope)
    assert scope['args'].human_verified_part_recovery
    monkeypatch.setattr(sys, 'argv', [c for c in sys.argv if c != '--unknown-verified-dynamic-evidence'])
    with pytest.raises(SystemExit):
        exec(code, dict(__file__=str(source), __name__='cli_test'))
    assert not (tmp_path/'native').exists()
