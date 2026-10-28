"""Frame conversion for a velocity-tracking legged base, not drive wheels.

The input is a checked fixed-frame (vx, vy, yaw-rate) command. Conversion
does not establish that a locomotion policy tracks that command accurately.
"""

import math

import numpy as np


def fixed_to_body_velocity(command, body_yaw_radians):
    """Rotate planar translation into the current body frame; keep yaw rate."""
    value = np.asarray(command, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("finite three-component fixed-frame velocity required")
    if isinstance(body_yaw_radians, bool) or not math.isfinite(body_yaw_radians):
        raise ValueError("finite measured body yaw required")
    c, s = math.cos(body_yaw_radians), math.sin(body_yaw_radians)
    return np.array([c*value[0]+s*value[1], -s*value[0]+c*value[1], value[2]])


def bounded_body_velocity(command, *, maximum_translation_mps=.25, maximum_yaw_rps=.8):
    """Reject out-of-domain commands rather than silently clipping a safe input."""
    value = np.asarray(command, dtype=float)
    limits = (maximum_translation_mps, maximum_yaw_rps)
    if (value.shape != (3,) or not np.isfinite(value).all()
        or any(isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in limits)):
        raise ValueError("finite velocity and positive configured limits required")
    if np.linalg.norm(value[:2]) > maximum_translation_mps+1e-9 or abs(value[2]) > maximum_yaw_rps+1e-9:
        raise ValueError("command exceeds the explicit legged-base experiment limits")
    return value.copy()
