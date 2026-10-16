"""Build static restaurant additions as new USD layers; never overwrite sources."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Process-local DLL lookup for the already installed Windows USD Exchange.
# Kit supplies its own paths; this also permits the ordinary existing Python.
_dll_directory = Path(sys.executable).parent / "Lib/site-packages/usd_exchange.libs"
if os.name == "nt" and _dll_directory.is_dir():
    _usd_dll_cookie = os.add_dll_directory(str(_dll_directory))
    from pxr import UsdLux  # noqa: F401 - load installed plugin dependencies

from robot.scene.restaurant_service_build import build_service_scene, convert_food_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=ROOT / "scenes/generated/restaurant_robot.usda")
    parser.add_argument("--convert-assets", action="store_true")
    args = parser.parse_args()
    if args.convert_assets:
        convert_food_assets()
    result = build_service_scene(args.output, base_scene=args.base)
    report = args.output.with_suffix(".json")
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
