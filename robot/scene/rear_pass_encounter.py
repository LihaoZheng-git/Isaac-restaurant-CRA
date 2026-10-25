"""Separate authored lower-aisle encounter; no physical evidence or safe-space input.

The longer pedestrian leg supplies an observable walking window at unchanged
speed. Robot waypoints are ordinary obstacle-unaware service-route context.
Original restaurant layers are never modified by selecting this session setup.
"""

import math

import numpy as np


OPEN_REAR_LAYOUT = "lower-aisle"
OPEN_ROBOT_START_WORLD = (27.0, 5.1, 0.07382)
OPEN_ROBOT_YAW = math.pi / 2
OPEN_ROBOT_ROUTE_WORLD = ((27.0, 6.2), (27.0, 7.1), (27.0, 7.9))
OPEN_ACTOR_START_WORLD = (25.8, 7.0, 0.0)
OPEN_ACTOR_ROUTE_WORLD = ((29.8, 7.0, 0.0), (29.8, 12.5, 0.0))


def validate_crossing_layout(layout, *, direction, extension, crossing_y):
    if layout not in {"legacy", OPEN_REAR_LAYOUT}:
        raise ValueError("unknown authored crossing layout")
    if layout == OPEN_REAR_LAYOUT and (
        direction != "left-to-right" or extension != 0.0 or crossing_y != 8.9
    ):
        raise ValueError("lower-aisle layout requires left-to-right and default legacy selectors")
    return layout == OPEN_REAR_LAYOUT


def open_robot_route_local_xy():
    return tuple(map(tuple, (np.array(OPEN_ROBOT_ROUTE_WORLD) - OPEN_ROBOT_START_WORLD[:2])
                     @ np.array([[0.0, -1.0], [1.0, 0.0]])))
