"""One crossing followed by departure; authored actor targets are never CRA inputs."""

import numpy as np


class DepartingPedestrianRoute:
    """Handoff from the crossing before braking; no loop or endpoint-stop claim."""

    route = ((29.8, 8.9, 0.0), (29.8, 12.5, 0.0))

    def __init__(self, *, crossing_direction="left-to-right"):
        if crossing_direction not in {"left-to-right", "right-to-left"}:
            raise ValueError("unsupported authored crossing direction")
        self.crossing_direction = crossing_direction
        if crossing_direction == "right-to-left":
            # Mirror only the actor fixture about the restaurant route x=28.8.
            # These targets are never estimator or robot-controller inputs.
            self.route = ((27.8, 8.9, 0.0), (27.8, 12.5, 0.0))
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
        if self.index != 0 or distance > .25:
            return False
        self.events.append(dict(time=float(now), actual_xyz=xyz.tolist(),
                                crossing_target=list(self.goal), distance_m=distance))
        self.index = 1
        return True
