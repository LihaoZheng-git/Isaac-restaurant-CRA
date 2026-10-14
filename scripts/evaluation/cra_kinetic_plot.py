"""Plot actual recorded kinetic estimates alongside independent motion truth."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads((args.source / "kinetic-fields.json").read_text())
    times, truths, speeds, energies, phis = [], [], [], [], []
    for record in records:
        truth = record["actor_monitor"]
        matches = [
            item
            for item in record["objects"]
            if np.linalg.norm(
                np.array(item["diagnostics"]["centroid_world_xy"])
                - truth["center_world_xy"]
            )
            < 0.4
        ]
        times.append(truth["elapsed_seconds"])
        truths.append(abs(truth["velocity_world_xy"][1]))
        item = matches[0] if len(matches) == 1 else None
        speeds.append(
            item["attributes"]["motion"]["relative_speed_mps"]["value"]
            if item
            else np.nan
        )
        energies.append(
            item["cra_control"]["cra"]["factors"]["kinetic"]["severity"]
            if item
            else np.nan
        )
        phis.append(item["cra_control"]["cra"]["phi"] if item else np.nan)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].plot(
        times,
        truths,
        color="black",
        linestyle="--",
        label="Authored speed: evaluation truth only",
    )
    axes[0].plot(
        times, speeds, color="#1769aa", label="LiDAR/robot relative speed estimate"
    )
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend(fontsize=8)
    axes[1].plot(times, energies, color="#ae5c00")
    axes[1].set_ylabel("Energy severity\n(upper-bound proxy)")
    axes[2].plot(times, phis, color="#7b3294")
    axes[2].set_ylabel("Computed CRA phi")
    axes[2].set_xlabel("Simulation seconds after actor motion enabled")
    for axis in axes:
        axis.grid(alpha=0.2)
        for boundary in (2, 5, 7, 10):
            axis.axvline(boundary, color="grey", alpha=0.3)
    fig.suptitle(
        "Stationary robot, moving sphere: continuous kinetic evidence\nMass/contact/plume retention and unavailable-factor priors remain labeled",
        fontsize=11,
    )
    fig.savefig(args.source / "kinetic-validation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
