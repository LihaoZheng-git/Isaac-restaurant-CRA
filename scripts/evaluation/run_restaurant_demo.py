"""Run one frozen restaurant demo gate, preserving every failed attempt."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            if not (gate["passed"] and execution["source_unchanged"]
                    and execution["passed"] and meta["source_revision"] == revision
                    and meta["case"] == case and meta["config_sha256"] == digest(config)):
                parser.error("failed/mismatched prerequisite")
            if case.endswith("-component"):
                reports = list((folder/"native").glob("*validation.json"))
                if not (folder/"native/human-sensor-validation.json").exists() or any(
                    not json.loads(p.read_text()).get("passed", False) for p in reports
                ):
                    parser.error("every human component/source/geometry validation must pass")
            prerequisites.append(dict(path=str(folder), report_sha256=digest(folder/filename)))
        command.append("--motion")
    output = args.output.resolve()
    output.mkdir(parents=True)
    command.extend(["--output", str(output/"native" if component else output)])
    # Freeze referenced generated scene layers as well as Python source.
    scene_key = "--scene" if component else "--restaurant-scene"
    scene = Path(command[command.index(scene_key)+1])
    manifest = json.loads(scene.with_suffix(".json").read_text())
    scene_hashes = {str(p): digest(p) for p in (scene, Path(manifest["additions"]))}
    scene_hashes.update(manifest["source_sha256"])
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
