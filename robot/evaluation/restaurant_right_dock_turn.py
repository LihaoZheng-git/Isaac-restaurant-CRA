"""Native right-table finishing turn with a zero-forward command."""

import math

import numpy as np


TURN_DEGREES = 30.
MAXIMUM_YAW_RATE = .30
YAW_DEADBAND = math.radians(.5)
MAXIMUM_DRIFT_M = .05
MINIMUM_CLEARANCE_M = .02


def execute_right_dock_turn(*, state, step, clearance, time, dt):
    """Turn relative to the settled approach heading and record the sweep."""
    initial = state()
    goal = initial[2] - math.radians(TURN_DEGREES)
    planned_clearance = min(clearance((*initial[:2], yaw))
                            for yaw in np.linspace(initial[2], goal, 121))
    rows = []
    failure = None
    settled = False
    for _ in range(math.ceil(6. / dt)):
        measured = state()
        error = math.remainder(measured[2] - goal, math.tau)
        if error <= YAW_DEADBAND:
            settled = True
            break
        if (planned_clearance <= MINIMUM_CLEARANCE_M or clearance(measured) <= MINIMUM_CLEARANCE_M
                or math.dist(initial[:2], measured[:2]) > MAXIMUM_DRIFT_M):
            failure = "turn_clearance_or_drift"
            break
        command = (0., -min(MAXIMUM_YAW_RATE, 2.2 * error))
        rows.append(dict(time_seconds=time(), state=measured, phase="rotation", command=command))
        step(command)
    if not settled and failure is None:
        failure = "turn_timeout"
    for _ in range(math.ceil(.75 / dt)):
        rows.append(dict(time_seconds=time(), state=state(), phase="settle", command=(0., 0.)))
        step((0., 0.))
    final = state()
    rows.append(dict(time_seconds=time(), state=final, phase="settle", command=(0., 0.)))
    angle = -math.degrees(math.remainder(final[2] - initial[2], math.tau))
    minimum = min(clearance(row["state"]) for row in rows)
    drift = max(math.dist(initial[:2], row["state"][:2]) for row in rows)
    return dict(passed=failure is None and abs(angle - TURN_DEGREES) <= 2.
                and minimum > MINIMUM_CLEARANCE_M and drift <= MAXIMUM_DRIFT_M,
                failure=failure, target_clockwise_degrees=TURN_DEGREES, measured_clockwise_degrees=angle,
                initial_state=initial, final_state=final, maximum_center_drift_m=drift,
                minimum_planned_padded_clearance_m=planned_clearance,
                minimum_measured_padded_clearance_m=minimum, samples=rows,
                scope="Zero-forward native dock turn from the measured approach heading using a separate geometry monitor.")
