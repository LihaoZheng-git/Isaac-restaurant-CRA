"""Load the audited official Isaac human and compose a separate static scene."""

import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
# Resolve before SimulationApp changes its working directory.
args.base, args.output = args.base.resolve(), args.output.resolve()
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
import omni.client
from robot.scene.isaac_restaurant_server import (
    ISAAC_HUMAN_URL,
    ISAAC_HUMAN_SHA256,
    build_isaac_server_scene,
    validate_isaac_human_bytes,
)

try:
    result, _, data = omni.client.read_file(ISAAC_HUMAN_URL)
    if result != omni.client.Result.OK:
        raise RuntimeError(
            "official Isaac human is inaccessible; no substitute selected"
        )
    validate_isaac_human_bytes(bytes(data))
    report = build_isaac_server_scene(args.output, args.base)
    report["asset_sha256"] = ISAAC_HUMAN_SHA256
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report), flush=True)
except Exception:
    traceback.print_exc()
    raise
finally:
    app.close()
