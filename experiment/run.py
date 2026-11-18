"""Launch the three-phase restaurant service experiment with the v2 settings."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiment.mission import PHASE_ORDER, ROOT, read_recipe, save


CONFIG = ROOT / "experiment/configs/mission.json"
# Public checkpoint names can also be supplied through the environment, so
# the two project-trained models need not be copied into the repository.
WEIGHTS = {
    "SAM2_CHECKPOINT": "sam2.1_t.pt",
    "VISUAL_MASS_CHECKPOINT": "visual-mass.pt",
    "CONTACT_SEGMENTATION_CHECKPOINT": "contact-segmentation.pt",
    "PERSON_SEGMENTATION_CHECKPOINT": "yolo11n-seg.pt",
}

V2_NATIVE_REVISION = "c5052395b825b71dd3b866d8bef634eb9983c49f"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_inputs(scene_root, model_root):
    commands, recipe_hashes = read_recipe(CONFIG, scene_root)
    model_paths = {}
    files = set()
    for variable, name in WEIGHTS.items():
        model_paths[variable] = Path(
            os.environ.get(variable, model_root / name)
        ).resolve()
        files.add(model_paths[variable])
    for command in commands.values():
        for flag in (
            "--restaurant-scene",
            "--restaurant-layout-survey",
            "--left-outbound-cast-manifest",
            "--right-crossing-cast-manifest",
        ):
            if flag in command:
                files.add(Path(command[command.index(flag) + 1]))
    left = commands["left"]
    scene = Path(left[left.index("--restaurant-scene") + 1])
    manifest_path = scene.with_suffix(".json")
    survey = Path(left[left.index("--restaurant-layout-survey") + 1])
    files.update((manifest_path, survey.parent / "authored-height-column-mask.png"))
    missing = sorted(str(path) for path in files if not path.is_file())
    if missing:
        raise ValueError("Missing experiment inputs:\n  " + "\n  ".join(missing))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files.add((scene_root / manifest["additions"]).resolve())
    expected = {
        (scene_root / key).resolve(): value
        for key, value in manifest["source_sha256"].items()
    }
    files.update(expected)
    fingerprints = {str(path): digest(path) for path in sorted(files)}
    for path, value in expected.items():
        if fingerprints[str(path)] != value:
            raise ValueError(
                f"Scene input differs from its construction manifest: {path}"
            )
    return recipe_hashes, model_paths, fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-root",
        type=Path,
        required=True,
        help="prepared restaurant scene bundle",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="directory holding the four perception checkpoints",
    )
    parser.add_argument(
        "--isaac-python",
        type=Path,
        required=True,
        help="Windows standalone Isaac Sim python.bat",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check",
        action="store_true",
        help="validate input paths, configuration, and scene fingerprints, then exit",
    )
    group.add_argument(
        "--output",
        type=Path,
        help="new directory for the native recording and run receipts",
    )
    args = parser.parse_args()
    scene_root, model_root, isaac_python = (
        p.resolve() for p in (args.scene_root, args.model_root, args.isaac_python)
    )
    if not isaac_python.is_file():
        parser.error(f"Isaac Sim launcher missing: {isaac_python}")
    try:
        recipes, models, files = prepare_inputs(scene_root, model_root)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    if args.check:
        print(
            f"Validated {len(recipes)} phase recipes and {len(files)} inputs. Isaac Sim was not started."
        )
        return 0

    output = args.output.resolve()
    if output.exists():
        parser.error(f"Choose a new output directory: {output}")
    command = [
        str(isaac_python),
        "-u",
        str(ROOT / "experiment/mission.py"),
        "--scene-root",
        str(scene_root),
        "--output",
        str(output),
    ]
    env = os.environ.copy()
    # Use the interpreter with the vision packages inside the Isaac environment.
    env.update({key: str(path) for key, path in models.items()})
    env.update(
        PYTHONEXE=f'"{sys.executable}"',
        CRA_SCENE_ROOT=str(scene_root),
        ISAAC_SIM_HEADLESS="1",
        ISAAC_SIM_USE_D3D12="1",
        ISAAC_RTX_HISTORY_TRANSFORM_COUNT="128",
        CRA_NATIVE_HUMAN="1",
        OPENBLAS_NUM_THREADS="16",
        OMP_NUM_THREADS="1",
    )
    env.setdefault("YOLO_DEVICE", "0")
    source_revision = None
    if (ROOT / ".git").exists():
        try:
            source_revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    sources = {
        p.relative_to(ROOT).as_posix(): digest(p)
        for name in ("experiment", "robot")
        for p in (ROOT / name).rglob("*.py")
    }
    for config_file in CONFIG.parent.glob("*.json"):
        sources[config_file.relative_to(ROOT).as_posix()] = digest(config_file)
    output.mkdir(parents=True)
    save(
        output / "invocation.json",
        dict(
            command=command,
            source_revision=source_revision,
            recorded_baseline_revision=V2_NATIVE_REVISION,
            source_sha256=sources,
            input_sha256=files,
            config_sha256=digest(CONFIG),
            requested_phases=PHASE_ORDER,
            referenced_config_sha256=recipes,
        ),
    )
    started = time.monotonic()
    # Kit may exit the child process from inside SimulationApp.close().
    traceback_seen = False
    with subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            traceback_seen |= line.startswith("Traceback (most recent call last):")
        code = process.wait()
    report = output / "report.json"
    result = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {}
    unchanged = all(
        (ROOT / path).is_file() and digest(ROOT / path) == value
        for path, value in sources.items()
    ) and all(
        Path(path).is_file() and digest(Path(path)) == value
        for path, value in files.items()
    )
    passed = bool(
        code == 0
        and not traceback_seen
        and unchanged
        and result.get("passed")
        and result.get("phase_recipe_sha256") == recipes
        and result.get("requested_phases") == list(PHASE_ORDER)
    )
    save(
        output / "execution.json",
        dict(
            launcher_exit_code=code,
            wall_seconds=time.monotonic() - started,
            traceback=traceback_seen,
            source_unchanged=unchanged,
            report_passed=result.get("passed"),
            passed=passed,
        ),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
