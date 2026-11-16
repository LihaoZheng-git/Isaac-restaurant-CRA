"""Independent zero-command tail scoring; never perception or controller input."""

import numpy as np


def summarize_post_goal_observation(braking, observation, *, goal_xy, goal_tolerance):
    """Preserve the original braking window and audit the entire extra trace.

    The caller has already reached the goal and closed planning/inference.
    Nothing here can turn failed arrival or rear passing into success.
    """
    first = np.asarray(braking['initial_state'], float)
    boundary = np.asarray(braking['final_state'], float)
    following = np.asarray(observation['initial_state'], float)
    goal = np.asarray(goal_xy, float)
    if (first.shape != (3,) or boundary.shape != (3,) or following.shape != (3,)
        or goal.shape != (2,) or not all(np.isfinite(x).all() for x in (first, boundary, following, goal))
        or not np.isfinite(goal_tolerance) or goal_tolerance <= 0
        or not np.allclose(boundary, following, rtol=0, atol=1e-9)):
        raise ValueError('original braking/tail boundary and finite goal required')
    records = list(braking['records']) + list(observation['records'])[1:]
    states = np.asarray([r['state'] for r in records], float)
    clearances = np.asarray([r['clearance_m'] for r in records], float)
    if (states.shape != (len(records), 3) or len(records) < 2
        or not np.isfinite(states).all() or not np.isfinite(clearances).all()
        or not np.allclose(states[0], first, rtol=0, atol=1e-9)
        or not np.allclose(states[-1], observation['final_state'], rtol=0, atol=1e-9)):
        raise ValueError('complete finite braking/tail samples required')
    distances = np.linalg.norm(states[:, :2] - first[:2], axis=1)
    goal_distances = np.linalg.norm(states[:, :2] - goal, axis=1)
    checks = dict(
        clearance_through_entire_zero_command_window=bool(clearances.min() > .02),
        original_start_total_drift_below_unchanged_stop_limit=bool(distances.max() < .05),
        goal_region_retained=bool(goal_distances.max() <= goal_tolerance),
    )
    return dict(passed=all(checks.values()), checks=checks,
        original_braking_drift_m=braking['translation_drift_m'],
        final_state=states[-1].tolist(),
        translation_drift_m=float(distances[-1]),
        maximum_total_translation_drift_m=float(distances.max()),
        minimum_clearance_m=float(clearances.min()),
        maximum_goal_distance_m=float(goal_distances.max()),
        final_goal_distance_m=float(goal_distances[-1]),
        original_braking_samples=len(braking['records']),
        additional_observation_samples=len(observation['records']),
        scope='Separate post-arrival zero-command observation; original braking window retained; no navigation pause, new physical evidence, field/source-clock renewal or new rear-pass acceptance')
