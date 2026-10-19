"""Run the fixed ORACLE field control audit; no Isaac Sim or wheel commands."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robot.evaluation.guidance_control_audit import (
    GuidanceAuditConfig,
    build_audit_field,
    run_control_audit,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--side", nargs="+", choices=("left", "right"), default=("left", "right")
    )
    parser.add_argument(
        "--kind",
        nargs="+",
        choices=("planar", "unicycle"),
        default=("planar", "unicycle"),
    )
    parser.add_argument("--maximum-seconds", type=float, default=40.0)
    parser.add_argument("--cbf-angular-weight", type=float, default=1.0)
    parser.add_argument("--solver-iterations", type=int, default=1000)
    parser.add_argument("--progress-seed-fraction", type=float, default=0.5)
    args = parser.parse_args()
    config = GuidanceAuditConfig(
        maximum_seconds=args.maximum_seconds,
        cbf_angular_weight=args.cbf_angular_weight,
        solver_iterations=args.solver_iterations,
        progress_seed_fraction=args.progress_seed_fraction,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise RuntimeError("retain previous attempts; choose a fresh output directory")
    manifest = dict(
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        configuration=asdict(config),
        sides=args.side,
        controllers=args.kind,
        nominal="same obstacle-unaware waypoint horizon, mapped into each command space",
        input_provenance="ORACLE disk geometry and phi; no physical estimator",
        fixed_lambda=1.25,
        nonzero_wheel_commands=0,
        working_tree_dirty=bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        audit_source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                ROOT / "robot/evaluation/guidance_control_audit.py",
                ROOT / "robot/safety/safe_sage_unicycle_mpc.py",
                ROOT / "robot/safety/safe_sage_planar_mpc.py",
            )
        },
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summaries, paths = [], {}
    for side in args.side:
        field = build_audit_field(side, config)
        digest = hashlib.sha256(
            field.h_full.values.tobytes()
            + b"".join(g.vectors_robot_xy.tobytes() for g in field.guidance_fields)
        ).hexdigest()
        for kind in args.kind:
            print(
                json.dumps({"starting": [side, kind], "field_sha256": digest}),
                flush=True,
            )
            report = run_control_audit(
                field, kind, config, progress=lambda p: print(json.dumps(p), flush=True)
            )
            report.update(passing_side=side, field_sha256=digest)
            (args.output / f"{side}-{kind}.json").write_text(
                json.dumps(report, indent=2)
            )
            summary = {
                k: v
                for k, v in report.items()
                if k not in {"records", "trajectory", "plans"}
            }
            summaries.append(summary)
            (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
            print(json.dumps(summary), flush=True)
            paths[side, kind] = np.asarray(report["trajectory"])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(args.side),
        figsize=(6 * len(args.side), 4.6),
        squeeze=False,
        layout="constrained",
    )
    for index, side in enumerate(args.side):
        ax = axes[0, index]
        ax.add_patch(plt.Circle((1.6, 0), 0.25, color="#777777", label="Oracle disk"))
        ax.plot([0, 3.3], [0, 0], "k--", alpha=0.5, label="Nominal waypoint route")
        for kind in args.kind:
            xy = paths[side, kind]
            row = next(
                s for s in summaries if s["passing_side"] == side and s["kind"] == kind
            )
            ax.plot(*xy[:, :2].T, label=kind + ": " + row["reason"])
            ax.scatter(*xy[-1, :2], marker="o" if row["passed"] else "x")
        ax.set(
            title=f"{side.title()} guidance, lambda 1.25",
            xlabel="Chassis x (m)",
            ylabel="Chassis y (m)",
            xlim=(-0.2, 3.5),
            ylim=(-1.6, 1.6),
            aspect="equal",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Fixed-field control isolation — ORACLE inputs, virtual motion only")
    fig.savefig(args.output / "trajectories.png", dpi=160)


if __name__ == "__main__":
    main()
