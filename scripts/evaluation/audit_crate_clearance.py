"""Measure crate clearance at recorded navigation and braking poses after an Isaac run."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robot.evaluation.simple_object_test import disk_footprint_clearance
from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint
from robot.scene.controlled_semantic_obstacle import (
    PlanarFootprint,
    WorldBounds3D,
    oriented_footprint_clearance_to_bounds,
)


# Physical Carter chassis used by the isolated native crate test, before padding.
CARTER = RectangularRobotFootprint(0.14, 0.587351, 0.2466, 0.2466, 0.05)


def rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def audit(source):
    def read(name):
        return json.loads((source / name).read_text(encoding="utf-8"))

    invocation = read("invocation.json")
    monitor = read("fixture-monitor.json")
    if (
        invocation["case"] != "crate-static"
        or invocation["stage"] != "motion"
        or [f["name"] for f in monitor["fixtures"]] != ["BrokenCrate"]
    ):
        raise ValueError("expected an isolated broken-crate motion trial")
    fixture = monitor["fixtures"][0]
    (prop,) = (
        p
        for p in read("restaurant-placement-monitor.json")["props"]
        if p["name"] == fixture["name"]
    )
    bounds = WorldBounds3D(tuple(prop["bounds_minimum"]), tuple(prop["bounds_maximum"]))
    report = read("report.json")
    commands = read("commands.json")
    trajectory = read("trajectory.json")
    braking = read("stopping-sweep.json")["records"]
    if len(commands) != len(trajectory) or not trajectory or not braking:
        raise ValueError("expected a paired command/pose history and a braking sweep")
    if [r["frame"] for r in braking] != list(range(len(braking))):
        raise ValueError("braking frames are incomplete")

    anchor = np.asarray(monitor["anchor_world_xy_yaw"])
    to_world = rotation(anchor[2])
    local_center = (np.asarray(fixture["center_world_xy"]) - anchor[:2]) @ to_world
    goal = np.asarray(report["goal_local_xy"])
    if (
        report["nominal_route_policy"]["configured_anchors_local_xy"] != [goal.tolist()]
        or np.linalg.norm(goal) == 0
    ):
        raise ValueError("expected a straight single-goal route from local (0, 0)")
    direction = (goal / np.linalg.norm(goal)) @ to_world.T
    lateral = np.array([-direction[1], direction[0]])
    box_xy = np.array(
        [
            (x, y)
            for x in (bounds.minimum[0], bounds.maximum[0])
            for y in (bounds.minimum[1], bounds.maximum[1])
        ]
    )
    along_box = (box_xy - anchor[:2]) @ direction
    padded = PlanarFootprint(
        *(
            extent + CARTER.safety_margin
            for extent in (
                CARTER.front_extent,
                CARTER.rear_extent,
                CARTER.left_extent,
                CARTER.right_extent,
            )
        )
    )
    body = np.array(
        [
            (x, y)
            for x in (-padded.rear_extent, padded.front_extent)
            for y in (-padded.right_extent, padded.left_extent)
        ]
    )

    rows = []
    for phase, poses in (
        ("navigation", trajectory),
        ("braking", [r["state"] for r in braking]),
    ):
        for i, pose in enumerate(poses):
            xy = np.asarray(pose[:2]) @ to_world.T + anchor[:2]
            yaw = pose[2] + anchor[2]
            along = float((xy - anchor[:2]) @ direction)
            rows.append(
                dict(
                    phase=phase,
                    index=i,
                    disk=disk_footprint_clearance(
                        pose,
                        center_xy=local_center,
                        radius_m=fixture["monitor_radius_m"],
                        footprint=CARTER,
                    ),
                    box=oriented_footprint_clearance_to_bounds(
                        (*xy, yaw), footprint=padded, obstacle_bounds=bounds
                    ),
                    lateral=float((xy - anchor[:2]) @ lateral),
                    beside_crate=bool(along_box.min() <= along <= along_box.max()),
                    entire_robot_past_crate=bool(
                        np.min(body @ rotation(yaw).T @ direction + along)
                        > along_box.max()
                    ),
                )
            )

    def minima(samples):
        return dict(
            poses=len(samples),
            minimum_disk_m=min(row["disk"] for row in samples),
            minimum_box_m=min(row["box"] for row in samples),
            disk_at_or_below_2cm=sum(row["disk"] <= 0.02 for row in samples),
            box_at_or_below_2cm=sum(row["box"] <= 0.02 for row in samples),
        )

    first_past = next((r for r in rows if r["entire_robot_past_crate"]), None)
    brake_start = len(trajectory)
    return dict(
        revision=invocation["source_revision"],
        reason=report["reason"],
        reached_goal=report["completed"],
        final_goal_error_m=math.dist(braking[-1]["state"][:2], goal),
        padded_chassis_m=asdict(padded),
        crate_bounds_world_m=asdict(bounds),
        disk_radius_m=fixture["monitor_radius_m"],
        navigation=minima(rows[:brake_start]),
        braking=minima(rows[brake_start:]),
        all_recorded=minima(rows),
        first_entire_robot_past_crate=first_past,
        reached_goal_with_entire_robot_past_crate=bool(
            report["completed"] and rows[-1]["entire_robot_past_crate"]
        ),
        max_abs_lateral_m=max(abs(r["lateral"]) for r in rows),
        max_abs_lateral_beside_crate_m=max(
            (abs(r["lateral"]) for r in rows if r["beside_crate"]), default=None
        ),
        last_issued=commands[-1],
        original_all_obstacle_minimum_before_braking_m=report[
            "minimum_clearance_before_stop_m"
        ],
        original_all_obstacle_minimum_during_braking_m=report[
            "minimum_clearance_during_stop_m"
        ],
        scope="Crate-only planar geometry at recorded poses. Braking includes the final control-tick pose. "
        "Measurements between poses and PhysX contact events are outside this audit.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.source.resolve()):
        parser.error("store the audit outside the frozen motion directory")
    result = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["all_recorded"], indent=2))


if __name__ == "__main__":
    main()
