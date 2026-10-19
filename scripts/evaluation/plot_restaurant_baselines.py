"""Plot actual matched runs, retaining failed paths and evidence limitations."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "geometric_velocity_cbf": "Geometric CBF",
    "fixed_class_safe_sage": "Fixed-class Safe-SAGE",
    "continuous_cra_safe_sage": "CRA + Safe-SAGE",
}
COLORS = dict(zip(LABELS, ("#6a7685", "#dc8a28", "#176cba"), strict=True))


def world_path(local, anchor):
    c, s = np.cos(anchor[2]), np.sin(anchor[2])
    return np.asarray(local)[:, :2] @ np.array([[c, s], [-s, c]]) + anchor[:2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.source / "comparison.json").read_text(encoding="utf-8"))
    cases = list(dict.fromkeys(row["case"] for row in data["trials"]))
    if not cases:
        raise ValueError("no executed trials to plot")
    fig = plt.figure(figsize=(max(12, 5 * len(cases)), 8), layout="constrained")
    layout = fig.add_gridspec(2, len(cases), height_ratios=(3, 1.7))
    table_rows = []
    for index, case in enumerate(cases):
        axis = fig.add_subplot(layout[0, index])
        selected = [r for r in data["trials"] if r["case"] == case]
        route_drawn = False
        for row in selected:
            source = Path(row["directory"])
            baseline = row["baseline"]
            if (source / "trajectory.json").is_file():
                trajectory = np.array(json.loads((source / "trajectory.json").read_text()))
                monitor = json.loads((source / "fixture-monitor.json").read_text())
                anchor = monitor["anchor_world_xy_yaw"]
                if len(trajectory):
                    path = world_path(trajectory, anchor)
                    axis.plot(path[:, 0], path[:, 1], color=COLORS[baseline], lw=2, label=LABELS[baseline])
                    axis.scatter(*path[-1], color=COLORS[baseline], marker="o" if row["passed"] else "x", s=55)
                route = row["configuration"].get("nominal_route_local_xy")
                if route and not route_drawn:
                    route = world_path([[0, 0], *route], anchor)
                    axis.plot(route[:, 0], route[:, 1], "k--", alpha=.45, label="Nominal waypoints")
                    route_drawn = True
                truth_path = source / "human-truth-evaluation-only.json"
                if truth_path.is_file():
                    truth = json.loads(truth_path.read_text())
                    xy = np.array([r["actor_xyz"][:2] for r in truth])
                    if len(xy):
                        axis.plot(xy[:, 0], xy[:, 1], ":", color=COLORS[baseline], alpha=.5)
            clearance = row["minimum_signed_monitor_clearance_m"]
            elapsed = row["navigation_simulation_seconds"]
            length = row["path_length_m"]
            coverage = row["perception"]
            table_rows.append([
                case.replace("_", " "), LABELS[baseline], "PASS" if row["passed"] else row["reason"],
                "--" if clearance is None else f"{clearance:.3f}",
                "--" if elapsed is None else f"{elapsed:.2f}",
                "--" if length is None else f"{length:.2f}",
                f"{coverage['accepted_instance_assessments']}/{coverage['object_assessments']}",
            ])
        axis.set(title=case.replace("_", " ").title(), xlabel="World x (m)", ylabel="World y (m)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=.2)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            axis.legend(unique.values(), unique.keys(), fontsize=8, loc="best")
    bottom = fig.add_subplot(layout[1, :])
    bottom.axis("off")
    table = bottom.table(cellText=table_rows,
        colLabels=["Case", "Method", "Full acceptance", "Min. clearance (m)", "Sim. time (s)", "Path (m)", "Fresh masks / assessments"],
        cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    incomplete = any(not cell["all_motion_reports"] for cell in data.get("matched_cells", []))
    fig.suptitle("Restaurant baseline comparison — " + ("incomplete pilot" if incomplete else "actual bounded prior-assisted trials"), fontsize=15)
    fig.supxlabel("Circles: full acceptance; crosses: failed endpoints. Dotted lines: independent human-root evaluation traces.\n"
        "Room/prop occupancy omitted from this trajectory plot. Mass/contact/fall/thermal priors retain provenance; no calibrated injury or real-time claim.", fontsize=9)
    output = args.source / "comparison-overview.png"
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
