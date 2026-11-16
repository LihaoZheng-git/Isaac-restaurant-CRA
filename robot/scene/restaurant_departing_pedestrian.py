"""One crossing followed by departure; authored actor targets are never CRA inputs."""

import math
import numpy as np


class DepartingPedestrianRoute:
    """Handoff from the crossing before braking; no loop or endpoint-stop claim."""

    route = ((29.8, 8.9, 0.0), (29.8, 12.5, 0.0))

    def __init__(self, *, crossing_direction="left-to-right", crossing_extension_m=0., crossing_y_m=8.9,
                 crossing_layout="legacy"):
        from robot.scene.rear_pass_encounter import validate_crossing_layout, OPEN_ACTOR_ROUTE_WORLD
        self.open_layout = validate_crossing_layout(crossing_layout, direction=crossing_direction,
            extension=crossing_extension_m, crossing_y=crossing_y_m)
        self.crossing_layout = crossing_layout
        if crossing_direction not in {"left-to-right", "right-to-left"}:
            raise ValueError("unsupported authored crossing direction")
        self.crossing_direction = crossing_direction
        if (isinstance(crossing_extension_m, bool) or not math.isfinite(crossing_extension_m)
            or not 0 <= crossing_extension_m <= .5):
            raise ValueError("separate authored crossing extension must be 0..0.5 m")
        self.crossing_extension_m = crossing_extension_m
        if isinstance(crossing_y_m, bool) or not math.isfinite(crossing_y_m) or not 8.4 <= crossing_y_m <= 8.9:
            raise ValueError("separate authored crossing line must be y=8.4..8.9")
        self.crossing_y_m = crossing_y_m
        if crossing_direction == "right-to-left":
            # Mirror only the actor fixture about the restaurant route x=28.8.
            # These targets are never estimator or robot-controller inputs.
            self.route = ((27.8, 8.9, 0.0), (27.8, 12.5, 0.0))
        if crossing_extension_m:
            delta = crossing_extension_m*(1 if crossing_direction=='left-to-right' else -1)
            self.route = tuple((x+delta,y,z) for x,y,z in self.route)
        self.route = ((self.route[0][0],crossing_y_m,0.),self.route[1])
        if self.open_layout:
            self.route = OPEN_ACTOR_ROUTE_WORLD
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
