from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from robot.safety.encounter_motion import (
    EncounterMotionClassifier,
    EncounterMotionConfig,
)


def update(
    c,
    t,
    *,
    velocity=(0.5, 0.0),
    ids=(7,),
    points=100,
    matched=True,
    goal=(0.0, 4.0),
    angle=0.0,
):
    v = np.array(velocity)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    tracks = tuple(
        NS(
            track_id=k,
            centroid_world_xy=rot @ (np.array([-0.8, 1.0]) + v * t),
            velocity_world_xy=rot @ v,
            point_count=points,
            matched_previous_track=matched,
            observation_count=10,
        )
        for k in ids
    )
    return c.update(
        NS(timestamp_seconds=t, tracked_obstacles=tracks),
        robot_xy=(0.0, 0.0),
        goal_xy=rot @ goal,
        human_track_ids=ids,
    )["tracks"]


@pytest.mark.parametrize(
    "velocity,state,direction",
    [
        ((0.5, 0.0), "crossing", None),
        ((-0.5, 0.0), "crossing", None),
        ((0.0, 0.5), "parallel", "same"),
        ((0.0, -0.5), "parallel", "opposite"),
        ((0.0, 0.0), "stationary", None),
    ],
)
@pytest.mark.parametrize("rotation", [0.0, 0.73])
def test_classifies_mirrored_parallel_and_stationary_in_world_route_frame(
    velocity, state, direction, rotation
):
    c = EncounterMotionClassifier()
    for t in (0.1, 0.2):
        assert (
            update(c, t, velocity=velocity, angle=rotation)[7]["classification"]
            == "uncertain"
        )
    row = update(c, 0.3, velocity=velocity, angle=rotation)[7]
    assert row["classification"] == state and row["parallel_direction"] == direction
    assert row["source_timestamp_seconds"] == 0.3


def test_hysteresis_and_turn_require_new_consistent_current_motion():
    c = EncounterMotionClassifier()
    for t in (0.1, 0.2, 0.3):
        update(c, t)
    # 55 degrees remains crossing after entry at90 degrees.
    v = (0.5 * np.sin(np.deg2rad(55)), 0.5 * np.cos(np.deg2rad(55)))
    assert update(c, 0.4, velocity=v)[7]["classification"] == "crossing"
    assert update(c, 0.5, velocity=(0.0, 0.5))[7]["classification"] == "uncertain"
    assert update(c, 0.6, velocity=(0.0, 0.5))[7]["classification"] == "uncertain"
    assert update(c, 0.7, velocity=(0.0, 0.5))[7]["classification"] == "parallel"


@pytest.mark.parametrize(
    "failure",
    ["gap", "missing", "new_id", "sparse", "identity", "speed", "goal", "route_turn"],
)
def test_invalid_or_changed_inputs_disable_policy_without_renewing_old_state(failure):
    c = EncounterMotionClassifier()
    for t in (0.1, 0.2, 0.3):
        update(c, t)
    options = {}
    t = 0.4
    if failure == "gap":
        t = 1.0
    if failure == "missing":
        assert update(c, 0.4, ids=()) == {}
        t = 0.5
    if failure == "new_id":
        options["ids"] = (8,)
    if failure == "sparse":
        options["points"] = 3
    if failure == "identity":
        options["matched"] = False
    if failure == "speed":
        options["velocity"] = (3.0, 0.0)
    if failure == "goal":
        options["goal"] = (0.0, 0.01)
    if failure == "route_turn":
        options["goal"] = (4.0, 0.0)
    row = next(iter(update(c, t, **options).values()))
    assert row["classification"] == "uncertain"
    assert row["source_timestamp_seconds"] == t


def test_source_order_bad_inputs_and_configuration_are_rejected():
    c = EncounterMotionClassifier()
    update(c, 0.1)
    with pytest.raises(ValueError):
        update(c, 0.1)
    with pytest.raises(ValueError):
        update(c, 0.2, ids=(7, 7))
    with pytest.raises(ValueError):
        update(c, 0.2, velocity=(np.nan, 0.0))
    with pytest.raises(ValueError):
        replace(EncounterMotionConfig(), parallel_enter_degrees=60)
    with pytest.raises(ValueError):
        replace(EncounterMotionConfig(), minimum_observations=True)
