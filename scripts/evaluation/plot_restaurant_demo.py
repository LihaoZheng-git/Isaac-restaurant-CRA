"""Independent synchronized trajectory plots; authored truth is evaluation only."""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output_directory
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout="constrained")
    synchronized = []
    names = {
        "static": "Service segment",
        "parallel": "Parallel walker",
        "crossing": "Perpendicular crossing",
    }
    for ax, (case, title) in zip(axes, names.items()):
        source = folder / (case + "-motion-v6")
        if not (source / "report.json").exists():
            ax.set_title(title + " | PENDING")
            ax.text(
                0.5, 0.5, "Awaiting verified trial", transform=ax.transAxes, ha="center"
            )
            ax.set_axis_off()
            continue
        report = json.loads((source / "report.json").read_text())
        execution = json.loads((source / "execution.json").read_text())
        accepted = (
            report["passed"] and execution["passed"] and execution["source_unchanged"]
        )
        y_upper = 11.0
        scene = json.loads(
            (ROOT / "scenes/generated/restaurant_demo_v6.json").read_text()
        )
        for prop in scene["props"]:
            lo, hi = (
                np.asarray(prop["bounds_minimum"]),
                np.asarray(prop["bounds_maximum"]),
            )
            ax.add_patch(
                Rectangle(
                    lo[:2],
                    *(hi - lo)[:2],
                    facecolor="#e3e5e7",
                    edgecolor="#adb2b7",
                    zorder=0,
                )
            )
        if case == "static":
            rows = json.loads((source / "service-video-epochs.json").read_text())[
                "frames"
            ]
            robot = np.asarray([r["robot_world_xy"] for r in rows])
            time = np.asarray([r["time"] for r in rows])
        else:
            rows = json.loads((source / "human-truth-evaluation-only.json").read_text())
            robot = np.asarray([r["robot_xy"] for r in rows])
            person = np.asarray([r["actor_xyz"][:2] for r in rows])
            time = np.asarray([r["time"] for r in rows])
            speed = np.linalg.norm(np.gradient(person, time, axis=0), axis=1)
            walking = np.asarray(
                [r["phase"] in ("move", "walk_away") for r in rows]
            ) & (speed >= 0.3)
            y_upper = max(y_upper, float(person[walking, 1].max()) + 0.2)
            ax.plot(
                person[walking, 0],
                person[walking, 1],
                color="#ba7026",
                lw=2,
                label="Human while walking",
            )
            separation = np.linalg.norm(robot - person, axis=1)
            eligible = np.flatnonzero(walking & (separation <= 2.0))
            if len(eligible):
                bent = eligible[np.abs(robot[eligible, 0] - 28.8) >= 0.2]
                index = (
                    (
                        bent[0]
                        if len(bent)
                        else eligible[np.argmax(np.abs(robot[eligible, 0] - 28.8))]
                    )
                    if case == "parallel"
                    else eligible[np.argmin(separation[eligible])]
                )
                ax.plot(
                    [robot[index, 0], person[index, 0]],
                    [robot[index, 1], person[index, 1]],
                    ":",
                    color="#444444",
                    lw=1.4,
                )
                ax.scatter(
                    *person[index], s=80, color="#ba7026", edgecolors="black", zorder=5
                )
                ax.scatter(
                    *robot[index], s=80, color="#1674a5", edgecolors="black", zorder=5
                )
                ax.annotate(
                    f"Same t={time[index]:.1f}s\ncenters {separation[index]:.2f}m",
                    (robot[index] + person[index]) / 2,
                    xytext=(8, 10),
                    textcoords="offset points",
                    fontsize=9,
                )
                synchronized.append(
                    dict(
                        case=case,
                        time_seconds=float(time[index]),
                        robot_world_xy=robot[index].tolist(),
                        human_world_xy=person[index].tolist(),
                        center_separation_m=float(separation[index]),
                        selection="first nearby 20cm bend (or largest if below threshold)"
                        if case == "parallel"
                        else "closest nearby walking centers",
                    )
                )
        ax.plot(
            robot[:, 0],
            robot[:, 1],
            color="#1674a5",
            lw=2.5,
            label="Executed robot path",
        )
        setup = json.loads((source / "restaurant-setup-validation.json").read_text())
        x, y, yaw = setup["configured_pose_world_xy_yaw"]
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        route = np.asarray(report["nominal_route_local_xy"]) @ rotation.T + [x, y]
        chord = np.vstack(([x, y], route))
        ax.plot(
            chord[:, 0],
            chord[:, 1],
            "--",
            color="#888888",
            lw=1.2,
            label="Nominal waypoints",
        )
        ax.scatter(
            *route[-1], marker="*", s=180, color="#287b54", zorder=6, label="Final goal"
        )
        ax.set_title(
            title + " | " + ("PASS" if accepted else "FAIL"), fontweight="bold"
        )
        ax.set(
            xlabel="World x (m)",
            ylabel="World y (m)",
            aspect="equal",
            xlim=(27.3, 31.2),
            ylim=(5.2, y_upper),
        )
        ax.grid(alpha=0.15)
        ax.text(
            0.02,
            0.02,
            f"{report['waypoints_reached']} waypoints | {report['navigation_simulation_seconds']:.1f}s\n"
            f"clearance {report['minimum_clearance_m']:.3f}m | stop drift {1000 * report['stop_drift_m']:.1f}mm",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
        )
        ax.legend(loc="lower left", bbox_to_anchor=(0, 0.15), fontsize=8)
    fig.suptitle(
        "First restaurant service demo: separate native navigation trials",
        fontsize=17,
        fontweight="bold",
    )
    fig.supxlabel(
        "Prior-assisted simulation. Marked robot/person positions share a timestamp; truth is evaluation-only. Not a continuous mission or real-time result.",
        fontsize=10,
    )
    fig.savefig(folder / "navigation-results.png", dpi=160)
    (folder / "synchronized-positions.json").write_text(
        json.dumps(synchronized, indent=2) + "\n"
    )
    bin_source = folder / "bin-summary-v4/results.json"
    if bin_source.exists():
        result = json.loads(bin_source.read_text())["trials"][0]
        names = ["potential", "kinetic", "contact", "thermal"]
        scores = [result["median_scores"][name] for name in names]
        figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
        bars = axis.bar(
            names, scores, color=["#ae8b4c", "#4c8aaf", "#4c8aaf", "#ae8b4c"]
        )
        axis.bar_label(bars, fmt="%.3f", padding=5)
        axis.set(ylim=(0, 1), ylabel="Median conservative CRA factor score")
        axis.set_title(
            f"Rounded bin: stationary component\n{result['accepted_rows']} accepted owned observations"
        )
        figure.supxlabel(
            "Potential / thermal: configured priors. Contact: visible geometry proxy.",
            fontsize=10,
        )
        figure.savefig(folder / "bin-summary-v4/factor-scores-demo.png", dpi=160)


if __name__ == "__main__":
    main()
