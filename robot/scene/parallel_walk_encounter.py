"""Authored short oncoming-aisle fixture; never perception/control evidence."""

import math

PARALLEL_ROBOT_START = (28.8, 7.0, .07382)
PARALLEL_ROBOT_YAW = math.pi / 2
PARALLEL_ROBOT_ROUTE = ((1.0, 0.0), (2.0, 0.0), (3.5, 0.0))
PARALLEL_ACTOR_START = (29.8, 10.5, 0.0)
PARALLEL_ACTOR_GOAL = (29.8, 7.0, 0.0)


def validate_parallel_scope(*, direction, extension, crossing_y, layout):
    if (direction != 'right-to-left' or extension != 0 or crossing_y != 8.9 or layout != 'legacy'):
        raise ValueError('parallel fixture requires unchanged legacy crossing selectors')
