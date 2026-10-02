"""Run repeated stopped-robot camera/LiDAR captures and controlled diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.evaluation.stationary_perception_benchmark import summarize_stationary_captures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--object-checkpoint", type=Path, required=True)
    parser.add_argument("--mass-checkpoint", type=Path, required=True)
    parser.add_argument("--contact-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-count", type=int, default=3)
    parser.add_argument("--device", default="0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--skip-controlled-lidar", action="store_true")
    parser.add_argument("--run-timeout-seconds", type=int, default=600)
    parser.add_argument("--fall-support-drop-m", type=float, default=0.80)
    return parser.parse_args()


def _run_script(isaac_python: Path, script: str, directory: Path, environment: dict[str, str], timeout: int):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "process.log").open("w", encoding="utf-8") as stream:
        try:
            completed = subprocess.run(
                [str(isaac_python), "-u", str(ROOT / script)],
                cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                check=False, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return 124
    return completed.returncode


def _read_payload(path: Path) -> dict[str, object]:
    def reject_nonfinite(value):
        raise ValueError(f"nonfinite diagnostic value: {value}")

    result = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
    if not isinstance(result, dict):
        raise ValueError("diagnostic payload must be an object")
    return result


def main() -> None:
    args = parse_args()
    for name in ("isaac_python", "object_checkpoint", "mass_checkpoint", "contact_checkpoint"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        setattr(args, name, path)
    if args.capture_count < 1 or args.run_timeout_seconds < 1:
        raise ValueError("capture-count and run-timeout-seconds must be positive")
    if not math.isfinite(args.fall_support_drop_m) or args.fall_support_drop_m <= 0:
        raise ValueError("fall-support-drop-m must be positive and finite")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"benchmark output directory already exists: {output}")
    output.mkdir(parents=True)
    run_id = "stationary-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    environment = dict(os.environ)
    environment.update({
        "YOLO_MODEL_PATH": str(args.object_checkpoint),
        "VISUAL_MASS_CHECKPOINT": str(args.mass_checkpoint),
        "CONTACT_SEGMENTATION_CHECKPOINT": str(args.contact_checkpoint),
        "YOLO_DEVICE": args.device, "FUSION_AUTOCLOSE": "1",
        "ISAAC_SIM_AUTOCLOSE": "1", "ISAAC_SIM_HEADLESS": "1" if args.headless else "0",
        "WARP_CACHE_PATH": str(output / "warp"), "CRA_BENCHMARK_RUN_ID": run_id,
        "CRA_FALL_SUPPORT_DROP_M": str(args.fall_support_drop_m),
    })
    environment.pop("HOT_SOUP_DETECTION_CHECKPOINT", None)
    captures = []
    process_failures = []
    for index in range(args.capture_count):
        capture_id = f"capture-{index + 1:03d}"
        directory = output / capture_id
        environment["CRA_BENCHMARK_CAPTURE_ID"] = capture_id
        environment["CRA_STATIONARY_BENCHMARK_DIR"] = str(directory)
        print(f"Running stationary restaurant {capture_id}...", flush=True)
        return_code = _run_script(
            args.isaac_python, "scripts/demos/camera_lidar_semantic_fusion_demo.py",
            directory, environment, args.run_timeout_seconds,
        )
        path = directory / "capture.json"
        # SimulationApp can terminate with zero after a Python error; the
        # structured artifact, not the process exit code alone, establishes success.
        if return_code or not path.is_file():
            process_failures.append(f"{capture_id}: no valid capture artifact; process exit={return_code}")
            print(process_failures[-1], flush=True)
            continue
        try:
            payload = _read_payload(path)
        except ValueError as error:
            process_failures.append(f"{capture_id}: invalid capture artifact: {error}")
            continue
        captures.append(payload)
        print(f"{capture_id}: capture_passed={payload['capture_passed']}", flush=True)
    summary = summarize_stationary_captures(tuple(captures), minimum_captures=args.capture_count)
    summary["dataflow_blockers"].extend(process_failures)
    summary["dataflow_passed"] = not summary["dataflow_blockers"]
    controlled = {}
    if not args.skip_controlled_lidar:
        environment.pop("CRA_STATIONARY_BENCHMARK_DIR", None)
        for name, script in (
            ("fall_height", "scripts/demos/lidar_fall_height_isaac_diagnostic.py"),
            ("object_motion", "scripts/demos/obstacle_tracking_demo.py"),
        ):
            directory = output / name
            environment["CRA_LIDAR_DIAGNOSTIC_DIR"] = str(directory)
            print(f"Running controlled {name} diagnostic with the robot stopped...", flush=True)
            return_code = _run_script(args.isaac_python, script, directory, environment, args.run_timeout_seconds)
            path = directory / "report.json"
            if return_code or not path.is_file():
                controlled[name] = {"passed": False, "blockers": [f"no valid diagnostic artifact; process exit={return_code}"]}
            else:
                try:
                    controlled[name] = _read_payload(path)
                except ValueError as error:
                    controlled[name] = {"passed": False, "blockers": [f"invalid diagnostic artifact: {error}"]}
            print(f"{name}: passed={controlled[name]['passed']}", flush=True)
    summary["run_id"] = run_id
    summary["controlled_lidar_components"] = controlled
    summary["capture_paths"] = [str(output / item["capture_id"] / "capture.json") for item in captures]
    summary["benchmark_execution_completed"] = len(captures) == args.capture_count and not process_failures
    summary["benchmark_passed"] = summary["dataflow_passed"] and all(item["passed"] for item in controlled.values())
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not summary["benchmark_passed"]:
        raise RuntimeError("stationary benchmark contains failed component checks; inspect summary.json")


if __name__ == "__main__":
    main()
