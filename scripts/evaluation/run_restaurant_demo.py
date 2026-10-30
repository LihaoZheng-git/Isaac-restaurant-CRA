"""Run one frozen restaurant demo gate, preserving every failed attempt."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = "scripts/evaluation/run_restaurant_demo.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prerequisite_matches(meta, execution, gate, case, config_hash, hashes, scene_hashes,
                         expected_command=None):
    """Require identical simulation sources even across a launcher-only commit."""
    def runtime(values):
        return {p: h for p, h in values.items() if p != LAUNCHER}
    if expected_command is None:
        configuration_matches = meta.get("config_sha256") == config_hash
    else:
        command = meta.get("command", [])
        configuration_matches = (command.count("--output") == 1
            and len(command) >= 2 and command[-2] == "--output"
            and command[:-2] == expected_command)
    return bool(gate.get("passed") and execution.get("passed")
                and execution.get("source_unchanged") and meta.get("case") == case
                and configuration_matches
                and runtime(meta.get("source_sha256", {})) == runtime(hashes)
                and meta.get("scene_sha256") == scene_hashes)


def component_reports_pass(folder):
    """Some legacy setup reports contain clearances instead of a passed flag."""
    native = folder/"native"
    if any(not (native/name).exists() for name in (
        "human-sensor-validation.json", "robot-setup-validation.json"
    )):
        return False
    for path in native.glob("*validation.json"):
        report = json.loads(path.read_text())
        if path.name == "robot-setup-validation.json":
            values = [report.get(k) for k in ("configured_clearance_m", "settled_clearance_m")]
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       and math.isfinite(v) and v > .02 for v in values):
                return False
        elif report.get("passed") is not True:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--stage", choices=("stationary", "motion", "component"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stationary-source", type=Path)
    parser.add_argument("--component-source", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    config = ROOT / "configs/restaurant_demo_v1.json"
    recipe = json.loads(config.read_text())
    if args.case not in recipe["commands"]:
        parser.error("unknown demo case")
    if args.output.exists():
        parser.error("output must be new; preserve previous trials")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        parser.error("commit the source before a native trial")
    hashes = {p.as_posix(): digest(p) for folder in ("robot", "scripts") for p in Path(folder).rglob("*.py")}
    prerequisites = []
    command = ["C:/isaacsim/python.bat", "-u", *recipe["commands"][args.case]]
    component = args.case.endswith("-component")
    if component != (args.stage == "component"):
        parser.error("component stage must use a component recipe")
    scene_key = "--scene" if component else "--restaurant-scene"
    scene = Path(command[command.index(scene_key)+1])
    manifest = json.loads(scene.with_suffix(".json").read_text())
    scene_hashes = {str(p): digest(p) for p in (scene, Path(manifest["additions"]))}
    scene_hashes.update({p: digest(Path(p)) for p in manifest["source_sha256"]})
    if any(scene_hashes[p] != h for p, h in manifest["source_sha256"].items()):
        parser.error("base scene changed since demo construction")
    if args.stage == "motion":
        if args.case not in {"static", "parallel", "crossing"} or args.stationary_source is None:
            parser.error("motion needs an eligible case and a same-source stationary gate")
        required = [(args.stationary_source, "stationary-gate.json", args.case)]
        if args.case != "static":
            if args.component_source is None:
                parser.error("human motion needs a new-scene component gate")
            required.append((args.component_source, "native/validation.json", args.case+"-component"))
        for folder, filename, case in required:
            meta = json.loads((folder/"invocation.json").read_text())
            execution = json.loads((folder/"execution.json").read_text())
            gate = json.loads((folder/filename).read_text())
            expected = ["C:/isaacsim/python.bat", "-u", *recipe["commands"][case]]
            if not prerequisite_matches(meta, execution, gate, case, digest(config), hashes,
                                         scene_hashes, expected_command=expected):
                parser.error("failed/mismatched prerequisite")
            if case.endswith("-component") and not component_reports_pass(folder):
                parser.error("every human component/source/geometry validation must pass")
            prerequisites.append(dict(path=str(folder), report_sha256=digest(folder/filename),
                source_revision=meta["source_revision"], same_revision=meta["source_revision"] == revision,
                runtime_scene_hashes_and_case_command_match=True,
                original_config_sha256=meta["config_sha256"],
                same_whole_config=meta["config_sha256"] == digest(config),
                non_runtime_source_exclusion=LAUNCHER))
        command.append("--motion")
    output = args.output.resolve()
    output.mkdir(parents=True)
    command.extend(["--output", str(output/"native" if component else output)])
    # Freeze referenced generated scene layers as well as Python source.
    meta = dict(source_revision=revision, source_sha256=hashes, scene_sha256=scene_hashes,
                config_sha256=digest(config), command=command, case=args.case, stage=args.stage,
                prerequisites=prerequisites, scope=recipe["scope"])
    (output/"invocation.json").write_text(json.dumps(meta, indent=2)+"\n")
    env = os.environ.copy()
    env["PYTHONEXE"] = '"'+sys.executable+'"'
    started = time.perf_counter()
    with (output/"launch.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    logfile = (output/"launch.log").read_text(encoding="utf-8", errors="replace")
    report_path = output/("native/validation.json" if component else
                          "report.json" if args.stage == "motion" else "stationary-gate.json")
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    unchanged = (subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == revision
                 and not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
                 and all(digest(Path(p)) == h for p, h in {**hashes, **scene_hashes}.items()))
    result = dict(launcher_exit_code=process.returncode, wall_seconds=time.perf_counter()-started,
                  traceback="Traceback (most recent call last)" in logfile,
                  source_unchanged=unchanged, report_passed=report.get("passed"),
                  passed=bool(process.returncode == 0 and report.get("passed") and unchanged
                              and "Traceback (most recent call last)" not in logfile))
    (output/"execution.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
