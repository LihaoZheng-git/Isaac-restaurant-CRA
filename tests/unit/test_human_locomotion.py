import copy
import math

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdSkel

from robot.evaluation.human_locomotion import summarize_human_locomotion
from robot.scene.restaurant_human_motion import (
    attach_moving_human_collider,
    configure_human_navmesh,
)


@pytest.mark.parametrize("trigger", [6.7, 8.5, float("nan")])
def test_native_crossing_rejects_out_of_scope_trigger_before_authoring(trigger):
    from robot.scene.restaurant_human_episode import RestaurantHumanEpisode

    with pytest.raises(ValueError, match="crossing trigger"):
        RestaurantHumanEpisode(
            None, mode="crossing", output=None, trigger_robot_y=trigger
        )


def test_standalone_crossing_accepts_approach_line_before_asset_authoring(monkeypatch):
    from robot.scene import restaurant_human_episode as module

    class ReachedAuthoring(Exception):
        pass

    def stop_authoring(*args, **kwargs):
        raise ReachedAuthoring

    monkeypatch.setattr(module, "prepare_native_restaurant_human", stop_authoring)
    with pytest.raises(ReachedAuthoring):
        module.RestaurantHumanEpisode(Usd.Stage.CreateInMemory(), mode="crossing",
            output=None, trigger_robot_y=8.4)


def test_continuous_crossing_keeps_legacy_trigger_scope():
    from robot.scene.restaurant_human_episode import RestaurantHumanEpisode

    with pytest.raises(ValueError, match="crossing trigger"):
        RestaurantHumanEpisode(None, mode="continuous-crossing", speed_mps=.5,
            output=None, trigger_robot_y=8.)


@pytest.mark.parametrize("speed", [0.49, 1.01, float("nan")])
def test_native_crossing_rejects_out_of_scope_speed_before_authoring(speed):
    from robot.scene.restaurant_human_episode import RestaurantHumanEpisode

    with pytest.raises(ValueError, match="walking speed"):
        RestaurantHumanEpisode(None, mode="crossing", output=None, speed_mps=speed)


def test_native_episode_abort_is_idempotent_and_preserves_completed_task():
    from robot.scene.restaurant_human_episode import RestaurantHumanEpisode

    class Agent:
        calls = 0

        def idle(self):
            self.calls += 1

    episode = RestaurantHumanEpisode.__new__(RestaurantHumanEpisode)
    episode.agent = Agent()
    episode.task = 3
    episode.task_succeeded = False
    episode.phase = "move"
    episode.abort()
    episode.abort()
    assert episode.phase == "aborted" and episode.agent.calls == 1
    episode.task_succeeded = True
    episode.phase = "stopped_after"
    episode.abort()
    assert episode.phase == "stopped_after" and episode.agent.calls == 1


def observations():
    rows = []
    for t in np.arange(0, 5, 0.05):
        phase = "idle_before" if t < 1 else "move" if t < 3 else "stopped_after"
        y = 0.7 * min(2, max(0, t - 1))
        step = 0.15 * math.sin(2 * math.pi * t * 2) if phase == "move" else 0.0
        rows.append(
            {
                "time": float(t),
                "phase": phase,
                "actor_xyz": [0, y, 0],
                "left_foot_xyz": [-0.1, y + step, 0.05],
                "right_foot_xyz": [0.1, y - step, 0.05],
                "robot_xy": [-2, 0],
                "collider_alignment_error_m": 0.001,
                "actor_layout_clearance_m": 0.3,
                "nonzero_wheel_command_count": 0,
            }
        )
    return rows


def test_articulated_arrival_speed_stop_and_parked_robot_gate():
    report = summarize_human_locomotion(
        observations(),
        goal_world_xyz=(0, 1.4, 0),
        requested_speed_mps=0.7,
        task_succeeded=True,
    )
    assert report["passed"]
    assert report["median_moving_speed_mps"] == pytest.approx(0.7)


@pytest.mark.parametrize(
    "problem",
    [
        "slide_without_animation",
        "wrong_speed",
        "no_arrival",
        "collider_left_behind",
        "robot_moved",
        "wheel_command",
        "collision",
        "not_stopped",
        "ungrounded",
    ],
)
def test_root_translation_or_clearance_alone_never_establishes_locomotion(problem):
    rows = copy.deepcopy(observations())
    for r in rows:
        if problem == "slide_without_animation":
            r["left_foot_xyz"] = [-0.1, r["actor_xyz"][1], 0.05]
            r["right_foot_xyz"] = [0.1, r["actor_xyz"][1], 0.05]
        elif problem == "wrong_speed":
            r["actor_xyz"][1] *= 0.1
        elif problem == "collider_left_behind":
            r["collider_alignment_error_m"] = 0.5
        elif problem == "robot_moved":
            r["robot_xy"] = [r["time"] * 0.1, 0]
        elif problem == "wheel_command":
            r["nonzero_wheel_command_count"] = 1
        elif problem == "collision":
            r["actor_layout_clearance_m"] = -0.01
        elif problem == "not_stopped" and r["phase"] == "stopped_after":
            r["actor_xyz"][1] += 0.1 * (r["time"] - 3)
        elif problem == "ungrounded":
            r["actor_xyz"][2] = 0.2
    report = summarize_human_locomotion(
        rows,
        goal_world_xyz=(0, 1.4, 0),
        requested_speed_mps=0.7,
        task_succeeded=problem != "no_arrival",
    )
    assert not report["passed"]


def test_duplicate_actor_epochs_are_invalid():
    rows = observations()
    rows[5]["time"] = rows[4]["time"]
    with pytest.raises(ValueError, match="epochs"):
        summarize_human_locomotion(
            rows,
            goal_world_xyz=(0, 1.4, 0),
            requested_speed_mps=0.7,
            task_succeeded=True,
        )


def test_nonfinite_clearance_cannot_be_ignored_by_minimum():
    rows = observations()
    rows[7]["actor_layout_clearance_m"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        summarize_human_locomotion(
            rows,
            goal_world_xyz=(0, 1.4, 0),
            requested_speed_mps=0.7,
            task_succeeded=True,
        )


def test_collider_is_kinematic_and_follows_moving_rig_without_mass_input():
    stage = Usd.Stage.CreateInMemory()
    rig = UsdSkel.Root.Define(stage, "/Actor/Rig")
    op = rig.AddTranslateOp()
    op.Set(Gf.Vec3d(3, 4, 0))
    collider = attach_moving_human_collider(stage, rig.GetPrim())
    assert collider.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.RigidBodyAPI(collider.GetPrim()).GetKinematicEnabledAttr().Get()
    assert not collider.GetPrim().HasAPI(UsdPhysics.MassAPI)
    assert tuple(
        collider.ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ).ExtractTranslation()
    ) == pytest.approx((3, 4, 0.85))
    op.Set(Gf.Vec3d(4, 5, 0))
    assert tuple(
        collider.ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ).ExtractTranslation()
    ) == pytest.approx((4, 5, 0.85))


def test_metre_navmesh_configuration_preserves_unrelated_metadata_and_disk(tmp_path):
    path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.GetRootLayer().customLayerData = {"user_note": "preserve"}
    stage.GetRootLayer().Save()
    before = path.read_bytes()
    values = configure_human_navmesh(stage)
    assert values["agentMinHeight"] == 1.7 and values["agentMaxRadius"] == 0.4
    assert (
        stage.GetRootLayer().customLayerData["user_note"] == "preserve"
        and path.read_bytes() == before
    )
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    with pytest.raises(ValueError, match="metre"):
        configure_human_navmesh(stage)
