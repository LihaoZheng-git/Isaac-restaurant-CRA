"""Independent parallel-actor and curved-avoidance scoring; no runtime inputs."""

import numpy as np


def validate_parallel_selection(records):
    """Audit real sensor selection records; no actor truth or command authority."""
    sources = set()
    checks = dict(explicit_oncoming_scope=True, current_supported_selection=True,
        original_visual_selection=True, original_selection_clock=True, away_side=True,
        selected_field_side_and_lambda=True)
    selected_rows = 0
    for r in records:
        a = r.get('motion_aware_passing') or r.get('diagnostics', {}).get('motion_aware_passing')
        if not a:
            continue
        checks['explicit_oncoming_scope'] &= a.get('encounter_geometry') == 'oncoming_parallel'
        preferences = a['preferences']
        if not preferences:
            continue
        selected_rows += 1
        source = r.get('time', r.get('sensor_timestamp'))
        sources.add(source)
        candidates = {int(c['track_id']): c for c in a['qualified_candidates']}
        checks['original_visual_selection'] &= a['original_selection_visual_person_verified'] is True
        started = a['encounter_selected_timestamp']
        checks['original_selection_clock'] &= (started is not None and started <= source
            <= started + a['configuration']['maximum_encounter_seconds'] + 1e-9)
        objects = {o['attributes']['track_id']: o for o in r['objects']}
        policies = r.get('applied_field_policies') or r.get('diagnostics', {}).get('applied_field_policies', ())
        policies = {p['track_or_group_id']: p for p in policies}
        for key, side in preferences.items():
            key = int(key)
            applied = policies.get(key, {})
            checks['selected_field_side_and_lambda'] &= applied.get('applied_passing_side') == side and applied.get('lambda') == 1.25
            if key not in objects or key not in candidates or candidates[key]['positive_lidar_point_count'] < 20:
                checks['current_supported_selection'] = False
                continue
            o = objects[key]
            v = [o['attributes']['motion'][k] for k in ('velocity_x_mps', 'velocity_y_mps')]
            current = all(x['value'] is not None and x['status'] in {'measured', 'inferred'}
                and abs(x['timestamp_seconds']-source) <= 1e-6 for x in v)
            checks['current_supported_selection'] &= current
            if not current:
                continue
            direction = np.array(a['route_direction_world_xy'])
            velocity = np.array([x['value'] for x in v])
            speed = np.linalg.norm(velocity)
            checks['current_supported_selection'] &= .2 <= speed <= 2. and -velocity @ direction >= np.cos(np.pi/6)*speed-1e-9
            rel = np.array(o['diagnostics']['centroid_world_xy']) - a['current_robot_world_xy']
            lateral = direction[0]*rel[1] - direction[1]*rel[0]
            checks['away_side'] &= abs(lateral) >= .15 and side == ('left' if lateral < 0 else 'right')
    checks['three_distinct_current_sources'] = len(sources) >= 3
    return dict(passed=all(checks.values()), checks={k: bool(v) for k, v in checks.items()},
        selected_rows=selected_rows, distinct_selected_sources=len(sources),
        scope='current visually associated oncoming selection/away tangent only; no rear passing, geometry/arrival/stopping or fresh four-factor claim')


def validate_parallel_actor(records, *, parked_robot):
    if len(records) < 30:
        return dict(passed=False, checks=dict(enough_actual_samples=False))
    times = np.array([r['time'] for r in records], float)
    xyz = np.array([r['actor_xyz'] for r in records], float)
    robot = np.array([r['robot_xy'] for r in records], float)
    if (not all(np.isfinite(a).all() for a in (times, xyz, robot)) or np.any(np.diff(times) <= 0)):
        raise ValueError('finite ordered independent parallel trace required')
    phase = np.array([r['phase'] for r in records])
    if not set(phase) <= {'idle_before', 'move', 'away_endpoint', 'aborted'}:
        raise ValueError('explicit parallel actor phases required')
    moving = phase == 'move'
    velocity = np.gradient(xyz[:, :2], times, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    steady = moving & (speed >= .3)
    median = float(np.median(speed[steady])) if steady.any() else None
    foot = np.array([np.array(r['left_foot_xyz']) - r['right_foot_xyz'] for r in records])
    geometry = np.array([[r['actor_layout_clearance_m'], r['collider_alignment_error_m']] for r in records])
    checks = dict(
        actual_oncoming_travel=xyz[0, 1] >= 10.2 and xyz[-1, 1] <= 7.3 and steady.sum() >= 10,
        bounded_parallel_direction=steady.any() and float(np.median(velocity[steady, 1])) < -.3
            and float(np.median(np.abs(velocity[steady, 0]))) <= .2,
        requested_half_metre_speed_observed=median is not None and .325 <= median <= .7,
        bounded_actor_region=bool(((xyz[:, 0] >= 29.3) & (xyz[:, 0] <= 30.3)
            & (xyz[:, 1] >= 6.5) & (xyz[:, 1] <= 11.)).all()),
        articulated_limbs=np.isfinite(foot).all() and moving.any() and np.max(np.ptp(foot[moving], axis=0)) >= .1,
        grounded_actor=np.max(np.abs(xyz[:, 2])) <= .05,
        independent_geometry=np.isfinite(geometry).all() and geometry[:, 0].min() > .02 and geometry[:, 1].max() <= .03,
        robot_zero_wheels=not parked_robot or (np.linalg.norm(robot-robot[0], axis=1).max() <= .02
            and all(r['nonzero_wheel_command_count'] == 0 for r in records)),
    )
    return dict(passed=all(checks.values()), checks={k: bool(v) for k, v in checks.items()},
        median_moving_speed_mps=median, minimum_actor_layout_clearance_m=float(geometry[:, 0].min()),
        maximum_collider_alignment_error_m=float(geometry[:, 1].max()),
        scope='authored oncoming aisle component; independent root/limbs/geometry only; away_endpoint is not a human stopping claim')


def validate_parallel_curve(records, truth, *, goal_world_xy):
    """A late curve or restored lambda alone cannot establish parallel avoidance."""
    actor = validate_parallel_actor(truth, parked_robot=False)
    if len(truth) < 30:
        return dict(passed=False, checks=dict(enough_actual_samples=False))
    t = np.array([r['time'] for r in truth], float)
    xy = np.array([r['robot_xy'] for r in truth], float)
    person = np.array([r['actor_xyz'][:2] for r in truth], float)
    goal = np.asarray(goal_world_xy, float)
    if goal.shape != (2,) or not np.isfinite(goal).all() or np.linalg.norm(goal-xy[0]) <= .15:
        raise ValueError('nontrivial configured robot route required')
    forward = (goal-xy[0])/np.linalg.norm(goal-xy[0])
    left = np.array([-forward[1], forward[0]])
    velocity = np.gradient(person, t, axis=0)
    walking = (np.array([r['phase'] for r in truth]) == 'move') & (np.linalg.norm(velocity, axis=1) >= .3)
    near = walking & (np.linalg.norm(xy-person, axis=1) <= 2.)
    excursion = float(np.max(np.abs((xy[near]-xy[0])@left))) if near.any() else None
    policy_sources = set()
    for r in records:
        for o in r['objects']:
            d = o['diagnostics']
            motion = o['attributes']['motion']
            source = r['sensor_timestamp']
            if not t[0] <= source <= t[-1] or d['policy_return_count'] <= 0:
                continue
            center = np.array([np.interp(source, t, person[:, j]) for j in range(2)])
            if np.linalg.norm(np.array(d['centroid_world_xy'])-center) > .45:
                continue
            values = [motion[k] for k in ('velocity_x_mps', 'velocity_y_mps')]
            if any(v['value'] is None or v['status'] not in {'measured', 'inferred'}
                or abs(v['timestamp_seconds']-source) > 1e-6 for v in values):
                continue
            v = np.array([v['value'] for v in values])
            speed = np.linalg.norm(v)
            if .2 <= speed <= 2. and v@forward < -np.cos(np.pi/6)*speed:
                policy_sources.add(source)
    policies = [p for r in records for p in r['diagnostics']['applied_field_policies']]
    checks = dict(actual_parallel_actor=actor['passed'], current_observed_oncoming_rows=len(policy_sources) >= 3,
        fixed_normal_guidance=bool(policies) and all(p['lambda'] == 1.25 for p in policies),
        curved_during_nearby_parallel_walk=excursion is not None and excursion >= .20,
        final_goal_rejoined=np.linalg.norm(xy[-1]-goal) <= .1)
    return dict(passed=all(checks.values()), checks={k: bool(v) for k, v in checks.items()},
        observed_parallel_rows=len(policy_sources), nearby_walking_excursion_m=excursion,
        required_excursion_m=.2, actor_component=actor,
        scope='independent parallel walking curve; actual arrival/stopping/clearance/containment/source gates remain separate; no enforced side, causal CRA or exact-paper claim')
