from dataclasses import replace

import numpy as np
import pytest

from robot.perception.human_body_envelope import HumanBodyEnvelope, HumanBodyEnvelopeConfig
from robot.perception.human_track_recovery import HumanTrackRecoveryConfig
from test_human_body_envelope import tracking, cue


def memory(**kwargs):
    result = HumanBodyEnvelope(HumanBodyEnvelopeConfig(**kwargs), recovery_config=HumanTrackRecoveryConfig())
    result.update(tracking(), cue(), camera_timestamp_seconds=1.)
    return result


def missing(time):
    return replace(tracking(time), tracked_obstacles=(), point_track_ids=np.empty(0, dtype=np.int64))


def test_native_failure_pattern_recovers_real_positive_geometry_without_clock_laundering():
    m = memory()
    for time in (1.1, 1.2):
        with pytest.raises(ValueError, match='current positive LiDAR track missing'):
            m.update(missing(time), {}, camera_timestamp_seconds=time, robot_world_xy=(0., 0.))
    result = m.update(tracking(1.3, (1.031143, 0.), 2, matched=True), {}, camera_timestamp_seconds=1.3)[0]
    assert result['original_person_camera_timestamp'] == 1.
    assert result['original_person_camera_frame_id'] == 11
    assert result['lidar_timestamp_seconds'] == 1.3
    assert result['current_support_count'] == 20
    assert result['radius_m'] == pytest.approx(.6)
    assert m.recovery_audit[0]['source_gap_seconds'] == pytest.approx(.3)
    assert m.recovery_audit[0]['centroid_prediction_residual_m'] == pytest.approx(.001143)


def test_unique_new_positive_id_reassociates_geometry_only():
    m = memory()
    t = tracking(1.2, (1.02, 0.), 1)
    t = replace(t, tracked_obstacles=(replace(t.tracked_obstacles[0], track_id=8),), point_track_ids=np.full(20, 8))
    row = m.update(t, {}, camera_timestamp_seconds=1.2)[0]
    assert row['track_id'] == 8 and row['envelope_track_id'] == 3_000_000_008
    assert set(m.identities) == {8}
    assert row['original_person_camera_timestamp'] == 1.
    assert m.recovery_audit[0]['changed_track_id']


@pytest.mark.parametrize('change', ['late', 'expired', 'competitor', 'jump', 'unmatched', 'count', 'speed'])
def test_recovery_preserves_failure_gates_transactionally(change):
    m = memory(maximum_identity_age_seconds=.25 if change == 'expired' else 30.)
    t = tracking(1.3, (1.03, 0.), 2, matched=True, other=change == 'competitor')
    if change == 'late':
        t = tracking(1.61, (1.061, 0.), 2, matched=True)
    if change == 'jump':
        t = tracking(1.3, (2., 0.), 2, matched=True)
    if change in {'unmatched', 'count', 'speed'}:
        track = replace(t.tracked_obstacles[0], **{
            'unmatched': dict(matched_previous_track=False),
            'count': dict(observation_count=1),
            'speed': dict(velocity_world_xy=(2., 0.))}[change])
        t = replace(t, tracked_obstacles=(track,))
    old = m.identities
    with pytest.raises(ValueError):
        m.update(t, {}, camera_timestamp_seconds=t.timestamp_seconds)
    assert m.identities is old and m.identities[7]['lidar_timestamp'] == 1.


def test_missing_far_person_is_rechecked_not_deleted_or_freshened():
    m = memory()
    assert m.update(missing(1.1), {}, camera_timestamp_seconds=1.1, robot_world_xy=(-5., 0.)) == []
    audit = m.recovery_audit[0]
    assert audit['clearance_m'] > 0 and audit['checked_horizon_seconds'] == 1.55
    assert m.identities[7]['lidar_timestamp'] == 1.
    with pytest.raises(ValueError, match='current positive LiDAR track missing'):
        m.update(missing(1.2), {}, camera_timestamp_seconds=1.2, robot_world_xy=(0., 0.))
    with pytest.raises(ValueError):
        m.update(missing(1.61), {}, camera_timestamp_seconds=1.61, robot_world_xy=(-5., 0.))


@pytest.mark.parametrize('kwargs', [dict(maximum_gap_seconds=.7), dict(maximum_gap_seconds=True),
    dict(checked_horizon_seconds=1.2), dict(maximum_robot_speed_mps=.2), dict(robot_radius_m=float('nan'))])
def test_invalid_recovery_scope_rejected(kwargs):
    with pytest.raises(ValueError):
        HumanTrackRecoveryConfig(**kwargs)


def test_far_gap_cannot_supply_fresh_or_retained_mass():
    from robot.perception.human_mass_identity_retention import HumanMassIdentityRetention
    m = memory()
    t = missing(1.1)
    m.update(t, {}, camera_timestamp_seconds=1.1, robot_world_xy=(-5., 0.))
    resolved, audit = HumanMassIdentityRetention().update(t, m, {}, timestamp_seconds=1.1, camera_frame_id=12)
    assert resolved == {} and audit['rows'] == []
    assert audit['missing_geometry_mass_not_resolved'] == [7]


@pytest.mark.parametrize('change', ['valid', 'missing_body', 'short_horizon', 'observed_only'])
def test_real_live_recovery_cli_before_isaac(monkeypatch, change):
    import ast
    from pathlib import Path
    import sys
    script = Path(__file__).resolve().parents[2] / 'scripts/demos/cra_live_object_isaac_demo.py'
    tree = ast.parse(script.read_text(encoding='utf-8'))
    split = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'args' for t in n.targets))
    guard = next(n for n in tree.body if isinstance(n, ast.If)
        and ast.unparse(n.test).startswith('args.human_track_recovery'))
    argv = [str(script), '--output', 'unused_recovery_test', '--human-track-recovery', '--mpc-horizon-steps', '6']
    if change != 'missing_body':
        argv += ['--human-body-envelope-radius', '.4']
    argv += {'short_horizon': ['--mpc-horizon-steps', '4'],
             'observed_only': ['--human-observed-occupancy-only']}.get(change, [])
    monkeypatch.setattr(sys, 'argv', argv)
    namespace = dict(__file__=str(script), __name__='recovery_cli_test')
    exec(compile(ast.Module(body=tree.body[:split+1], type_ignores=[]), str(script), 'exec'), namespace)
    code = compile(ast.Module(body=[guard], type_ignores=[]), str(script), 'exec')
    if change == 'valid':
        exec(code, namespace)
    else:
        with pytest.raises(SystemExit):
            exec(code, namespace)
