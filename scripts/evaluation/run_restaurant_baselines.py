"""Run/resume a prespecified bounded comparison. Never starts the long mission."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robot.evaluation.restaurant_comparison import (
    comparison_plan, comparison_summary, contract_fingerprint, expected_stationary_rejection,
    extract_trial, trial_options,
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "study"), default="pilot")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--continue-after-stationary-rejection", action="store_true",
                        help="retain a validated zero-wheel component rejection as an end-to-end failure, then try the next independent trial")
    parser.add_argument("--case", action="append", help="execute selected cases from the unchanged manifest")
    parser.add_argument("--baseline", action="append", help="execute selected baselines from the unchanged manifest")
    parser.add_argument("--isaac-python", type=Path, default=Path("C:/isaacsim/python.bat"))
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    trials = comparison_plan(phase=args.phase, repetitions=args.repetitions)
    options = {t["case"]: trial_options(t) for t in trials}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    manifest = {"schema_version": 1, "phase": args.phase, "trials": trials,
                "case_options": options, "code_revision": revision, "working_tree_dirty": dirty,
                "continue_after_stationary_rejection": args.continue_after_stationary_rejection,
                "case_contract_sha256": {k: contract_fingerprint(v) for k, v in options.items()},
                "scope": "short restaurant baselines only; labeled factor/adult/body-contact priors; no long service demonstration",
                "study_design": "nominal static/standing, exogenous crossing at 7/8/9 seconds; timing variants replace new placement changes; actor path unchanged",
                "fixed_class_policy_sha256": hashlib.sha256((ROOT / "configs/safe_sage_paper_baseline.json").read_bytes()).hexdigest()}
    path = args.output / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError("existing manifest differs; use a new directory rather than overwriting trials")
    else:
        save(path, manifest)
    if args.execute and dirty:
        raise RuntimeError("commit validated code before comparative collection")
    if args.execute and not args.isaac_python.is_file():
        raise FileNotFoundError(args.isaac_python)
    rows = []
    for trial in trials:
        directory = args.output / trial["case"] / f"repeat-{trial['repeat']:02d}" / trial["baseline"]
        done = directory / "execution.json"
        selected = (not args.case or trial["case"] in args.case) and (not args.baseline or trial["baseline"] in args.baseline)
        if args.execute and selected and not done.exists() and not (args.output / "STOP").exists():
            current_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            current_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
            if current_revision != revision or current_dirty:
                (args.output / "STOP").write_text("Source revision changed during collection; remaining trials were not started.\n", encoding="utf-8")
                raise RuntimeError("frozen source changed; retain completed trials and use a new output root")
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "stationary-fields.json").exists() or (directory / "live-fields.jsonl").exists():
                raise RuntimeError(f"unfinished trial artifacts retained at {directory}; use a new output root for a retry")
            command = [str(args.isaac_python), "-u", "scripts/demos/cra_live_object_isaac_demo.py",
                       "--output", str(directory), "--experiment-baseline", trial["baseline"],
                       *options[trial["case"]], "--motion"]
            print(json.dumps({"starting": trial, "directory": str(directory)}), flush=True)
            save(directory / "invocation.json", {"command": command, "code_revision": revision,
                "contract_sha256": contract_fingerprint(options[trial["case"]]),
                "started_utc": datetime.now(timezone.utc).isoformat()})
            environment = os.environ.copy()
            environment["PYTHONEXE"] = f'"{sys.executable}"'
            with (directory / "isaac.log").open("w", encoding="utf-8") as stream:
                process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                                         stderr=subprocess.STDOUT, check=False)
            # Isaac's batch launcher may return zero after a script exception.
            log = (directory / "isaac.log").read_text(encoding="utf-8", errors="replace")
            gate_path = directory / "stationary-gate.json"
            gate = json.loads(gate_path.read_text()) if gate_path.is_file() else {}
            save(done, {"launcher_returncode": process.returncode,
                        "traceback_present": "Traceback (most recent call last)" in log,
                        "expected_stationary_rejection": expected_stationary_rejection(gate, log),
                        "motion_report_available": (directory / "report.json").is_file(),
                        "ended_utc": datetime.now(timezone.utc).isoformat()})
            row = extract_trial(directory, trial)
            print(json.dumps({"finished": trial, "passed": row["passed"], "reason": row["reason"]}), flush=True)
        if done.exists():
            row = extract_trial(directory, trial)
            row["execution"] = json.loads(done.read_text(encoding="utf-8"))
            if row["passed"] and (row["execution"]["traceback_present"] or row["execution"]["launcher_returncode"] != 0):
                row["passed"] = False
                row["execution_validation_error"] = "report pass contradicted by process/log failure"
            rows.append(row)
            if args.execute and selected and not row["report_available"] and not (
                args.continue_after_stationary_rejection
                and row["execution"].get("expected_stationary_rejection", False)
            ):
                # A controller timeout is an outcome; failure before motion is
                # a setup/perception blocker, not another controller result.
                (args.output / "STOP").write_text(
                    f"Missing motion report in {directory}; retained for investigation.\n",
                    encoding="utf-8",
                )
        save(args.output / "comparison.json", comparison_summary(rows))
    print(json.dumps({"planned_trials": len(trials), "executed_trials": len(rows),
                      "passed": sum(r["passed"] for r in rows), "manifest": str(path)}), flush=True)


if __name__ == "__main__":
    main()
