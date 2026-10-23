from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from robot.safety.motion_aware_passing import (
    MotionAwarePassing,
    MotionAwarePassingConfig,
)
from robot.safety.safe_sage_guidance_policy import PassingSide


def sample(t, *, sign=1, ids=(7,), rotate=0.0, speed=0.5):
    c, s = np.cos(rotate), np.sin(rotate)
    rot = np.array([[c, -s], [s, c]])
    tracks = [
        NS(
            track_id=key,
            centroid_world_xy=tuple(rot @ np.array([sign * (-0.8 + speed * t), 0.8])),
            velocity_world_xy=tuple(rot @ np.array([sign * speed, 0.0])),
            point_count=100,
            observation_count=10,
            matched_previous_track=True,
        )
        for key in ids
    ]
    return NS(timestamp_seconds=t, tracked_obstacles=tuple(tracks)), rot


def step(policy, t, **options):
    tracking, rot = sample(t, **options)
    return policy.update(
        tracking,
        robot_xy=(0, 0),
        robot_velocity_xy=rot @ np.array([0, 0.2]),
        goal_xy=rot @ np.array([0, 4]),
    )


@pytest.mark.parametrize(
    "sign,expected", [(1, PassingSide.LEFT), (-1, PassingSide.RIGHT)]
)
@pytest.mark.parametrize("rotation", [0.0, 0.8, -2.0])
def test_crossing_selects_trailing_tangent_in_any_world_orientation(
    sign, expected, rotation
):
    p = MotionAwarePassing()
    for t in (0.1, 0.2):
        assert not step(p, t, sign=sign, rotate=rotation)[0]
    sides, audit = step(p, 0.3, sign=sign, rotate=rotation)
    assert sides == {7: expected}
    assert audit["qualified_candidates"][0]["predicted_separation_m"] < 1.2
    _, held = step(p, 0.4, sign=sign, rotate=rotation)
    assert held["encounter_selected_timestamp"] == 0.3


def test_stationary_history_cannot_count_as_three_moving_observations():
    p = MotionAwarePassing()
    for t in (0.1, 0.2, 0.3, 0.4):
        assert not step(p, t, speed=0)[0]
    assert not step(p, 0.5, speed=0.25)[0]
    assert not step(p, 0.6, speed=0.25)[0]
    assert step(p, 0.7, speed=0.25)[0]


def test_multiple_crossings_and_source_gaps_cannot_select_or_renew_preference():
    p = MotionAwarePassing()
    for t in (0.1, 0.2, 0.3):
        sides, a = step(p, t, ids=(7, 8))
    assert not sides and a["decision"] == "ambiguous_multiple_crossings"
    p = MotionAwarePassing()
    for t in (0.1, 0.2, 0.3):
        step(p, t)
    assert not step(p, 0.8)[0]
    assert p.encounter is None
    with pytest.raises(ValueError, match="increase"):
        step(p, 0.8)


def test_missing_identity_releases_without_transferring_to_new_track():
    p = MotionAwarePassing()
    for t in (0.1, 0.2, 0.3):
        step(p, t)
    assert not step(p, 0.4, ids=(8,))[0]
    assert p.encounter is None


def test_encounter_expiry_keeps_original_selection_time():
    p = MotionAwarePassing(
        replace(MotionAwarePassingConfig(), maximum_encounter_seconds=0.25)
    )
    for t in (0.1, 0.2, 0.3, 0.4, 0.5):
        step(p, t)
    assert p.encounter[2] == 0.3
    assert not step(p, 0.6)[0]


def test_invalid_geometry_and_duplicate_identity_are_rejected():
    p = MotionAwarePassing()
    with pytest.raises(ValueError, match="duplicate"):
        step(p, 0.1, ids=(7, 7))
    with pytest.raises(ValueError, match="finite"):
        step(p, 0.1, speed=float("nan"))


def test_configured_nominal_segment_keeps_crossing_selection_near_intermediate_waypoint():
    p=MotionAwarePassing()
    legacy=MotionAwarePassing()
    for t in (.1,.2,.3):
        tracking,_=sample(t)
        args=dict(robot_xy=(0.,0.),robot_velocity_xy=(0.,0.),goal_xy=(0.,.01))
        selected,audit=p.update(tracking,nominal_route_direction_world_xy=(0.,1.),**args)
        before,_=legacy.update(tracking,**args)
    assert selected and not before
    assert audit['encounter_selected_timestamp']==.3
    assert audit['route_direction_basis']=='configured_obstacle_unaware_nominal_route_segment'
