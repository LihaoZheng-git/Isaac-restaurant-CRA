"""Build a separate static-server restaurant layer using the audited local asset."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
dll = Path(sys.executable).parent / "Lib/site-packages/usd_exchange.libs"
if os.name == "nt" and dll.is_dir():
    cookie = os.add_dll_directory(str(dll))
    from pxr import UsdLux  # noqa: F401
from robot.scene.restaurant_server import build_server_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_server_scene(args.output, args.base)
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
