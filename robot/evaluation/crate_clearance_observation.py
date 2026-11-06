"""Measure the static crate benchmark after navigation from recorded robot poses."""

import math


def world_pose_from_local(pose, anchor):
    """Invert the live runner's planar transformation into its settled robot frame."""
    x, y, yaw = pose
    ax, ay, heading = anchor
    c, s = math.cos(heading), math.sin(heading)
    return (ax + c * x - s * y, ay + s * x + c * y, heading + yaw)


def summarize_clearance_samples(samples, threshold_m=0.02):
    """Report every sampled crossing of the original strict benchmark threshold."""
    crossings = []
    below = False
    count_at_or_below = 0
    for row in samples:
        clearance = row['clearance_m']
        if not math.isfinite(clearance):
            raise ValueError('independent clearance sample must be finite')
        now_below = clearance <= threshold_m
        count_at_or_below += int(now_below)
        if now_below != below:
            crossings.append(dict(row, transition='entered' if now_below else 'exited'))
        below = now_below
    return dict(
        mode='post_navigation_observation',
        threshold_m=threshold_m,
        samples=len(samples),
        samples_at_or_below_threshold=count_at_or_below,
        minimum_clearance_m=min(row['clearance_m'] for row in samples),
        legacy_clearance_passed=count_at_or_below == 0,
        crossings=crossings,
        scope='Original fixture disk and layout clearance evaluated at sampled poses. '
              'Each crossing time is the first sample on the new side of the threshold.',
    )
