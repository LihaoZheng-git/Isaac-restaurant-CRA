"""Convert licensed user-downloaded glTF assets with installed Isaac.

No downloads, installers, factor labels or wheel commands. Preserve original
geometry/materials; scene placement and scale are authored in separate layers.
"""

import asyncio
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets/restaurant/sketchfab_contact_props"


def audited_jobs(names=("recycle_bin", "crates")):
    jobs = []
    for name in names:
        folder = ASSETS / name
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["license"] != "CC-BY-4.0":
            raise ValueError("unexpected source license")
        for row in manifest["files"]:
            source = (folder / row["path"]).resolve()
            if (not source.is_relative_to(folder.resolve())
                    or hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]):
                raise ValueError("source fingerprint mismatch")
        target = folder / "converted.usdc"
        if target.exists():
            raise ValueError("preserve existing converted asset")
        jobs.append((folder / "source/scene.gltf", target))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", choices=("recycle_bin", "crates", "cleaver", "broken_plate", "vases"),
                        default=["recycle_bin", "crates"])
    args = parser.parse_args()
    if len(set(args.assets)) != len(args.assets):
        parser.error("duplicate asset names")
    jobs = audited_jobs(args.assets)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    from isaacsim.core.experimental.utils.app import enable_extension

    enable_extension("omni.kit.asset_converter")
    import omni.kit.asset_converter as converter

    async def convert():
        for source, target in jobs:
            context = converter.AssetConverterContext()
            context.ignore_materials = False
            context.use_meter_as_world_unit = True
            task = converter.get_instance().create_converter_task(
                str(source), str(target), lambda _a, _b: None, context)
            if not await task.wait_until_finished():
                raise RuntimeError(f"installed glTF conversion failed: {source}")
            print(json.dumps({"converted": str(target),
                              "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}), flush=True)
    try:
        asyncio.get_event_loop().run_until_complete(convert())
    finally:
        app.close()


if __name__ == "__main__":
    main()
