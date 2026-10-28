"""Independent root-state evaluation for a bounded locomotion interface bench.

These poses are simulation evaluation truth, never CRA or navigation inputs.
This bench does not establish perception or obstacle-avoidance acceptance.
"""

import math

import numpy as np


def root_observation(*, time, phase_elapsed, phase, command_body, position_world,
                     roll_pitch_yaw, velocity_body):
    """Copy typed native vectors into finite strict-JSON evaluation samples."""
    result = dict(time=float(time), phase_elapsed=float(phase_elapsed), phase=phase)
    if not isinstance(phase, str) or not phase or not all(math.isfinite(result[k]) for k in ('time', 'phase_elapsed')):
        raise ValueError('finite source epoch and explicit phase required')
    for key, value in dict(command_body=command_body, position_world=position_world,
                          roll_pitch_yaw=roll_pitch_yaw, velocity_body=velocity_body).items():
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError('finite three-component native root vectors required')
        result[key] = vector.tolist()
    return result


def _height_limits(minimum, maximum):
    if (any(isinstance(v, bool) or not math.isfinite(v) for v in (minimum, maximum))
        or not minimum < maximum):
        raise ValueError('explicit ordered finite platform root-height bounds required')
    return minimum, maximum


def summarize_standing(rows, *, minimum_base_height_m=.3, maximum_base_height_m=.8):
    low, high = _height_limits(minimum_base_height_m, maximum_base_height_m)
    if len(rows) < 30:
        raise ValueError("at least thirty independent standing observations required")
    t = np.array([r["time"] for r in rows], float)
    p = np.array([r["position_world"] for r in rows], float)
    rpy = np.array([r["roll_pitch_yaw"] for r in rows], float)
    commands = np.array([r["command_body"] for r in rows], float)
    if (p.shape != (len(t), 3) or rpy.shape != p.shape or commands.shape != p.shape
        or not all(np.isfinite(a).all() for a in (t, p, rpy, commands))
        or np.any(np.diff(t) <= 0) or np.max(np.diff(t)) > .05+1e-9):
        raise ValueError("finite ordered root observations with gaps <=.05s required")
    drift = float(np.max(np.linalg.norm(p[:, :2]-p[0, :2], axis=1)))
    yaw = float(np.max(np.abs(np.unwrap(rpy[:, 2])-rpy[0, 2])))
    tilt = float(np.max(np.abs(rpy[:, :2])))
    checks = dict(zero_velocity_commands=bool(np.all(commands == 0)),
        at_least_three_seconds=bool(t[-1]-t[0] >= 3.-1e-9),
        bounded_standing_translation=drift <= .02,
        bounded_standing_yaw=yaw <= .03,
        upright=tilt <= math.radians(10),
        bounded_base_height=bool(np.all((p[:, 2] >= low) & (p[:, 2] <= high))))
    return dict(passed=all(checks.values()), checks=checks, frames=len(t),
        maximum_xy_drift_m=drift, maximum_yaw_drift_rad=yaw, maximum_tilt_rad=tilt,
        root_height_bounds_m=[low, high],
        scope="freely standing flat-ground zero-command policy bench; root truth is evaluation only; no sensor/CRA/navigation acceptance")


def summarize_tracking(rows, *, axes=('forward', 'lateral', 'yaw', 'combined'),
                       minimum_base_height_m=.3, maximum_base_height_m=.8):
    """Score settled tracking and each stop without hiding braking displacement."""
    low, high = _height_limits(minimum_base_height_m, maximum_base_height_m)
    if not axes or len(set(axes)) != len(axes) or not set(axes) <= {'forward', 'lateral', 'yaw', 'combined'}:
        raise ValueError('explicit unique supported command axes required')
    phases = list(dict.fromkeys(r["phase"] for r in rows))
    results = []
    for phase in phases:
        group = [r for r in rows if r["phase"] == phase]
        if not phase.startswith(("move_", "stop_")):
            continue
        elapsed = np.array([r["phase_elapsed"] for r in group], float)
        measured = np.array([r["velocity_body"] for r in group], float)
        command = np.array([r["command_body"] for r in group], float)
        position = np.array([r["position_world"] for r in group], float)
        angles = np.array([r["roll_pitch_yaw"] for r in group], float)
        if (len(group) < 30 or measured.shape != (len(group), 3) or command.shape != measured.shape
            or position.shape != measured.shape or angles.shape != measured.shape
            or np.any(np.diff(elapsed) <= 0) or np.max(np.diff(elapsed)) > .05+1e-9
            or not all(np.isfinite(a).all() for a in (elapsed, measured, command, position, angles))):
            raise ValueError("thirty finite samples required for each tracking phase")
        settled = elapsed >= 1.
        if settled.sum() < 20:
            raise ValueError("at least twenty settled samples required")
        error = measured[settled]-command[settled]
        xy_error = float(np.sqrt(np.mean(np.sum(error[:, :2]**2, axis=1))))
        yaw_error = float(np.sqrt(np.mean(error[:, 2]**2)))
        checks = dict(translation_tracking=xy_error <= .08, yaw_tracking=yaw_error <= .15,
            upright=bool(np.max(np.abs(angles[:, :2])) <= math.radians(10)),
            bounded_base_height=bool(np.all((position[:, 2] >= low) & (position[:, 2] <= high))),
            constant_phase_command=bool(np.all(command == command[0])),
            phase_command_active=bool(np.any(command[0] != 0) if phase.startswith('move_') else np.all(command == 0)))
        result = dict(phase=phase, translation_rmse_mps=xy_error, yaw_rmse_rps=yaw_error)
        if phase.startswith("stop_"):
            # From the first zero-command sample, not after an unreported settling period.
            result["maximum_stop_displacement_m"] = float(np.max(np.linalg.norm(position[:, :2]-position[0, :2], axis=1)))
            checks["original_two_cm_stop_gate"] = result["maximum_stop_displacement_m"] <= .02
        result.update(checks=checks, passed=all(checks.values()))
        results.append(result)
    if not results:
        raise ValueError("explicit movement and stopping phases required")
    required = {prefix+axis for prefix in ('move_', 'stop_') for axis in axes}
    complete = set(phases)-{'settle', 'standing'} == required
    return dict(passed=complete and all(r["passed"] for r in results), phases=results,
        complete_command_suite=complete,
        evaluated_axes=list(axes), root_height_bounds_m=[low, high],
        scope="bounded open-space command tracking only; no navigation, perception, ideal-velocity realization or causal platform benefit claim")
