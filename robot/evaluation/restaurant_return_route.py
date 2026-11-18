"""Nominal guides for the isolated left-table-to-pickup drive."""

import math

from robot.evaluation.restaurant_right_route import PICKUP


# Post-brake chassis pose from the accepted full left delivery (2026-11-12).
START = (4.43006038665771, 14.9323539733887, 0.767863046264878)
DINER_END = (0.9593145251274109, 7.9729204177856445, 0.)
PICKUP_POSITION_TOLERANCE_M = .04
PICKUP_HEADING_TOLERANCE_RAD = math.radians(5.)
TURN_GUIDE_COUNT = 5
TURN_LOOKAHEAD_M = 1.4
RETURN_GUIDES = (
    (6.00, 15.65),
    (6.90, 15.65),
    (7.00, 14.20),
    (5.85, 13.40),
    (3.40, 12.95),
    (2.05, 10.50),
    (2.05, 8.00),
    (1.85, 6.45),
    (3.70, 4.65),
    (5.80, 4.03),
    (7.20, 3.95),
    (11.20, 3.95),
    (23.40, 4.10),
    (24.90, 4.30),
    PICKUP[:2],
)


def return_route_local():
    x, y, yaw = START
    c, s = math.cos(yaw), math.sin(yaw)
    return tuple((c * (gx - x) + s * (gy - y), -s * (gx - x) + c * (gy - y))
                 for gx, gy in RETURN_GUIDES)
