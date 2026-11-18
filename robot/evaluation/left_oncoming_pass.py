"""Post-drive scoring for the northbound left-aisle encounter."""

import math

import numpy as np

from robot.evaluation.crate_clearance_observation import world_pose_from_local
from robot.evaluation.simple_object_test import disk_footprint_clearance


def summarize_left_oncoming_pass(
    samples,
    actor_records,
    *,
    anchor_world_xy_yaw,
    navigation_start_seconds,
    nominal_world_x,
    footprint,
    actor_radius_m,
    contact_report,
    minimum_clearance_m,
    west_side_diner=False,
):
    """Pair native robot/actor poses from the same physics epochs."""
    navigation = [row for row in samples if row["phase"] == "navigation"]
    times = np.asarray([row["time"] for row in actor_records], dtype=float)
    epochs = navigation_start_seconds + np.asarray(
        [row["time_seconds"] for row in navigation]
    )
    indices = np.searchsorted(times, epochs - 1e-7)
    paired = bool(
        len(epochs)
        and np.all(indices < len(times))
        and np.all(np.abs(times[indices] - epochs) <= 1e-7)
    )
    if not paired:
        result = dict(passed=False, checks={"same_epoch_robot_and_actor_poses": False})
        if west_side_diner:
            result["delivery_encounter_passed"] = False
        return result

    robot = np.asarray(
        [
            world_pose_from_local(row["local_state"], anchor_world_xy_yaw)
            for row in navigation
        ]
    )
    actor = np.asarray([actor_records[index]["actor_xyz"][:2] for index in indices])
    distances = np.linalg.norm(robot[:, :2] - actor, axis=1)
    moving = np.asarray([actor_records[index]["phase"] == "move" for index in indices])
    nearby = moving & (distances <= 2.0)
    order = robot[:, 1] - actor[:, 1]
    nearby_ahead = np.flatnonzero(nearby & (order <= -0.25))
    nearby_behind = np.flatnonzero(nearby & (order >= 0.25))
    passed_while_walking = bool(
        len(nearby_ahead) and len(nearby_behind) and nearby_ahead[0] < nearby_behind[-1]
    )
    robot_progress = (
        float(robot[nearby_behind[-1], 1] - robot[nearby_ahead[0], 1])
        if passed_while_walking
        else 0.0
    )
    lateral_offset = robot[nearby, 0] - nominal_world_x
    away_curve = float(np.max(lateral_offset if west_side_diner else -lateral_offset)) if lateral_offset.size else None
    curve_key = "curved_to_east_of_guide" if west_side_diner else "curved_to_west_of_guide"
    human_clearances = [
        disk_footprint_clearance(
            pose, center_xy=xy, radius_m=actor_radius_m, footprint=footprint
        )
        for pose, xy in zip(robot, actor)
    ]
    # The padded rear, not only the robot center, must be beyond the adult.
    yaw = float(robot[-1, 2])
    s, c = math.sin(yaw), math.cos(yaw)
    margin = footprint.safety_margin
    rear_y = (
        robot[-1, 1]
        + min(
            (footprint.front_extent + margin) * s, -(footprint.rear_extent + margin) * s
        )
        + min(
            (footprint.left_extent + margin) * c, -(footprint.right_extent + margin) * c
        )
    )
    rear_clearance = float(rear_y - actor[-1, 1] - actor_radius_m)
    floor = "/World/Environment/BuildingShell/Default_Floor_758"
    phases = contact_report["phases"]
    floor_only = (
        contact_report["installed"]
        and contact_report["error"] is None
        and all(
            phases[phase]["bin_contact_reported"] is False
            and set(phases[phase]["robot_environment_headers"]) == {floor}
            for phase in ("navigation", "braking")
        )
    )
    checks = dict(
        same_epoch_robot_and_actor_poses=True,
        passed_while_actor_walked=passed_while_walking,
        robot_advanced_through_encounter=robot_progress >= 0.15,
        **{curve_key: away_curve is not None and away_curve >= 0.20},
        padded_rear_cleared_actor=rear_clearance > 0.0,
        padded_human_and_layout_clearance=min(human_clearances) > 0.02
        and minimum_clearance_m > 0.02,
        navigation_and_braking_floor_only_contacts=floor_only,
    )
    if west_side_diner:
        moving_indices = np.flatnonzero(nearby)
        nearest = moving_indices[np.argmin(distances[moving_indices])] if moving_indices.size else None
        checks["diner_west_at_closest_moving_approach"] = bool(
            nearest is not None and actor[nearest, 0] < robot[nearest, 0]
        )
    result = dict(
        passed=all(checks.values()),
        checks=checks,
        **{("peak_eastward_curve_m" if west_side_diner else "peak_westward_curve_m"): away_curve},
        minimum_padded_human_clearance_m=float(min(human_clearances)),
        minimum_center_separation_m=float(min(distances)),
        robot_progress_through_nearby_pass_m=robot_progress,
        final_padded_rear_gap_m=rear_clearance,
        closest_approach_sim_seconds=float(
            navigation[int(np.argmin(distances))]["time_seconds"]
        ),
        scope="native pose, human collider and contact observations for post-drive scoring",
    )
    if west_side_diner:
        # A yield can let the diner pass while Carter is stationary.
        result["delivery_encounter_passed"] = all(
            value for key, value in checks.items()
            if key not in (curve_key, "robot_advanced_through_encounter")
        )
    return result
