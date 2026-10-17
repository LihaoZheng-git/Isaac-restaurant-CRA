"""Convert audited CC0 FBX with the installed Isaac converter; bake static idle.

No installer, dependency upgrades or downloaded executable code are used.
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists() or args.work.exists():
    parser.error("conversion requires fresh work/output paths")
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
from isaacsim.core.experimental.utils.app import enable_extension

enable_extension("omni.kit.asset_converter")
import omni.kit.asset_converter as converter
from robot.scene.restaurant_server import (
    SERVER_ASSET_ROOT,
    audit_server_sources,
    bake_server_asset,
)


async def main():
    audit_server_sources()
    args.work.mkdir(parents=True)
    for name, source in (
        ("body", "Model/characterMedium.fbx"),
        ("idle", "Animations/idle.fbx"),
    ):
        context = converter.AssetConverterContext()
        context.ignore_materials = False
        context.use_meter_as_world_unit = True
        task = converter.get_instance().create_converter_task(
            str(SERVER_ASSET_ROOT / "source" / source),
            str(args.work / (name + ".usd")),
            lambda _a, _b: None,
            context,
        )
        if not await task.wait_until_finished():
            raise RuntimeError("installed FBX converter failed")
    report = bake_server_asset(
        args.work / "body.usd", args.work / "idle.usd", args.output
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report), flush=True)


try:
    asyncio.get_event_loop().run_until_complete(main())
finally:
    app.close()
