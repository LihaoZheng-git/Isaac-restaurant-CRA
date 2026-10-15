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
        xy = np.array(fixture["offset_xy"])
        if "anchor_world_xy_yaw" in monitor:
            x, y, yaw = monitor["anchor_world_xy_yaw"]
            c, s = np.cos(yaw), np.sin(yaw)
            xy = (np.array(fixture["center_world_xy"]) - [x, y]) @ np.array([[c, -s], [s, c]])
        axis.add_patch(
            Circle(xy, fixture["radius_m"], color=colors[fixture["name"]], alpha=0.8)
        )
        if fixture["drop_m"]:
            axis.add_patch(
                Rectangle(np.array(xy) - 0.35, 0.7, 0.7, fill=False, edgecolor="grey")
            )
        axis.annotate(
            fixture["name"], np.array(xy) + [0, 0.25], ha="center", fontsize=9
        )
    goal_xy = report.get("goal_local_xy", [3.5 if report["fixture_layout"] == "three" else 2.5, 0])
    goal = goal_xy[0]
    axis.scatter([0, goal], [0, goal_xy[1]], color=["black", "#237742"], marker="x", s=60)
    axis.annotate("start", (0, 0), xytext=(-10, 12), textcoords="offset points")
    axis.annotate("waypoint", (goal, 0), xytext=(0, 12), textcoords="offset points")
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
        "Fixture geometry is oracle evaluation metadata; retained properties and unavailable factors use labeled priors.",
        fontsize=8,
    )
    fig.savefig(args.source / "navigation-validation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
