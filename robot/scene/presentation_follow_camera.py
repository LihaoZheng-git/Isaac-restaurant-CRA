"""Pure render-only camera pose smoothing; no sensor or control dependency."""

import math
import numpy as np


def follow_camera_pose(previous, actual, dt):
    old, pose = np.asarray(previous, float), np.asarray(actual, float)
    if old.shape != (3,) or pose.shape != (3,) or not np.isfinite([*old, *pose, dt]).all() or dt < 0:
        raise ValueError("camera pose/time must be finite, forward-time xy-yaw")
    alpha = 1-math.exp(-dt/.35)
    yaw_delta = math.atan2(math.sin(pose[2]-old[2]), math.cos(pose[2]-old[2]))
    smooth = old + alpha*np.array([pose[0]-old[0], pose[1]-old[1], yaw_delta])
    direction = np.array([math.cos(smooth[2]), math.sin(smooth[2])])
    eye = (* (smooth[:2]-1.8*direction), 2.5)
    target = (* (smooth[:2]+.8*direction), .55)
    return tuple(smooth), eye, target
