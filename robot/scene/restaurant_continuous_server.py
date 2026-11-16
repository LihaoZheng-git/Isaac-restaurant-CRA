"""Explicit actor-only circuit; authored waypoints never enter robot control."""

import numpy as np


CONTINUOUS_SERVER_ROUTE = (
    (28.0, 8.9, 0.0),
    (28.0, 11.4, 0.0),
    (29.8, 11.4, 0.0),
    (29.8, 8.9, 0.0),
)
UPPER_CONTINUATION_ROUTE = (
    (28.0, 8.9, 0.0),
    (28.0, 11.4, 0.0),
    (29.8, 11.4, 0.0),
    (29.8, 12.6, 0.0),
    (28.0, 12.6, 0.0),
)


class ContinuousServerRoute:
    """Request the next leg before braking; actual readbacks decide progress."""

    def __init__(self, *, upper_continuation=False):
        if not isinstance(upper_continuation, bool):
            raise TypeError("upper continuation opt-in must be boolean")
        self.route = (
            UPPER_CONTINUATION_ROUTE if upper_continuation else CONTINUOUS_SERVER_ROUTE
        )
        self.loop_start = 1 if upper_continuation else 0
        self.index = 0
        self.events = []

    @property
    def goal(self):
        return self.route[self.index]

    def advance(self, xyz, now):
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all() or not np.isfinite(now):
            raise ValueError("finite actual actor position/time required")
        distance = float(np.linalg.norm(xyz[:2] - self.goal[:2]))
        if distance > 0.25:
            return False
        self.events.append(
            {
                "time": now,
                "goal": list(self.goal),
                "actual_xyz": xyz.tolist(),
                "distance_m": distance,
            }
        )
        self.index += 1
        if self.index == len(self.route):
            self.index = self.loop_start
        return True
