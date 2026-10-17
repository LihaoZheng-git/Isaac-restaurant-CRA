"""Plot recorded CRA navigation with independently authored fixture monitors."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    trajectory = np.array(json.loads((args.source / "trajectory.json").read_text()))
    report = json.loads((args.source / "report.json").read_text())
    monitor = json.loads((args.source / "fixture-monitor.json").read_text())
    fixtures = monitor["fixtures"]
    # The fixture offsets are authored evaluation coordinates. The saved path
    # is chassis-center position in the anchored local frame, approximately the
    # configured world axes; do not use this plot to reconstruct sensor geometry.
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color="#1769aa",
        label="Recorded chassis-center trajectory",
    )
    colors = {"ground": "#399ddd", "raised": "#de8041", "plume": "#60b465"}
    for fixture in fixtures:
        if "anchor_world_xy_yaw" in monitor:
            x, y, yaw = monitor["anchor_world_xy_yaw"]
            c, s = np.cos(yaw), np.sin(yaw)
            xy = (np.array(fixture["center_world_xy"]) - [x, y]) @ np.array(
                [[c, -s], [s, c]]
            )
        else:
            xy = np.array(fixture["offset_xy"])
        radius = fixture.get("monitor_radius_m", fixture.get("radius_m"))
        axis.add_patch(
            Circle(xy, radius, color=colors.get(fixture["name"], "#de8041"), alpha=0.8)
        )
        if fixture["drop_m"]:
            axis.add_patch(
                Rectangle(np.array(xy) - 0.35, 0.7, 0.7, fill=False, edgecolor="grey")
            )
        axis.annotate(
            "actor at start" if report.get("moving_actor") else fixture["name"],
            np.array(xy) + [0, 0.25], ha="center", fontsize=9
        )
    goal_xy = report.get(
        "goal_local_xy", [3.5 if report["fixture_layout"] == "three" else 2.5, 0]
    )
    goal = goal_xy[0]
    route = np.array([(0, 0), *report.get("nominal_route_local_xy", [goal_xy])])
    axis.plot(
        route[:, 0],
        route[:, 1],
        ":",
        color="grey",
        linewidth=1,
        label="Obstacle-unaware waypoint route",
    )
    axis.scatter(route[1:, 0], route[1:, 1], marker="x", color="#237742", s=40)
    if report.get("moving_actor"):
        if (args.source / "live-fields.jsonl").exists():
            with (args.source / "live-fields.jsonl").open(encoding="utf-8") as stream:
                actor_xy = np.array(
                    [
                        json.loads(line)["actor_monitor"]["center_world_xy"]
                        for line in stream
                    ]
                )
        else:
            actor_xy = np.array(
                [
                    r["actor_monitor"]["center_world_xy"]
                    for r in json.loads((args.source / "live-fields.json").read_text())
                ]
            )
        if "anchor_world_xy_yaw" not in monitor:
            raise ValueError("moving-actor plot requires the recorded anchor")
        actor_xy = (actor_xy - [x, y]) @ np.array([[c, -s], [s, c]])
        axis.plot(
            actor_xy[:, 0],
            actor_xy[:, 1],
            "--",
            color="#de8041",
            label="Authored actor path (evaluation only)",
        )
        axis.add_patch(Circle(actor_xy[-1], fixtures[0]["radius_m"],
                              color="#de8041", alpha=.25))
        axis.annotate("actor at final sample", actor_xy[-1] + [.23, 0],
                      ha="left", va="center", fontsize=9)
    axis.scatter(
        [0, goal], [0, goal_xy[1]], color=["black", "#237742"], marker="x", s=60
    )
    axis.annotate("start", (0, 0), xytext=(-10, 12), textcoords="offset points")
    axis.annotate("final waypoint", (goal, goal_xy[1]), xytext=(-5, -18), ha="right", textcoords="offset points")
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)
    axis.set_xlabel("Anchored local x (m)")
    axis.set_ylabel("Anchored local y (m)")
    axis.legend(loc="lower left", fontsize=8)
    axis.set_title(
        f"Live CRA/Safe-SAGE: {report['reason']}\n{report['live_field_count']} fields; lambda=1.25; pacing={report['simulation_rate_limit']} simulation s/wall s",
        fontsize=11,
    )
    fig.supxlabel(
        "Fixture geometry is oracle evaluation metadata; retained properties and missing factors use labeled priors.\nRoom occupancy participates in control but is omitted from this plot.",
        fontsize=8,
    )
    fig.savefig(args.source / "navigation-validation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
