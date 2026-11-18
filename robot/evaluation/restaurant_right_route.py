"""Obstacle-unaware nominal guides for the right dining room trials."""

import math


PICKUP = (26.0, 4.30, 0.0)
RIGHT_GUIDES = (
    (27.40, 5.20),
    (28.65, 7.40),
    (28.80, 9.50),
    (29.75, 13.20),
    (29.75, 15.10),
    (29.80, 16.50),
    (29.75, 19.90),
)
TABLE_FRONT_DOCK = (30.25, 18.66)
TABLE_FRONT_APPROACH_INDEX = len(RIGHT_GUIDES) - 2
TABLE_FRONT_LOOKAHEAD_M = .55
TABLE_FRONT_GOAL_TOLERANCE_M = .15


def right_route(case, *, table_front=False, approach_probe=False):
    if approach_probe and not table_front:
        raise ValueError("The approach probe requires the table-front dock")
    if table_front:
        if case != "delivery":
            raise ValueError("The table-front dock requires the right delivery route")
        approach = ((29.85, 17.35), TABLE_FRONT_DOCK)
        if approach_probe:
            return (*RIGHT_GUIDES[-2], math.pi / 2), approach
        return PICKUP, (*RIGHT_GUIDES[:-1], *approach)
    if case == "boy":
        return PICKUP, RIGHT_GUIDES[:3]
    if case == "trolley":
        return (28.80, 11.0, math.pi / 2), RIGHT_GUIDES[3:6]
    if case == "delivery":
        return PICKUP, RIGHT_GUIDES
    raise ValueError(f"Unknown right-side trial {case!r}")


def right_route_local(case, *, table_front=False, approach_probe=False):
    (x, y, yaw), guides = right_route(case, table_front=table_front, approach_probe=approach_probe)
    c, s = math.cos(yaw), math.sin(yaw)
    return tuple((c * (gx - x) + s * (gy - y), -s * (gx - x) + c * (gy - y))
                 for gx, gy in guides)
