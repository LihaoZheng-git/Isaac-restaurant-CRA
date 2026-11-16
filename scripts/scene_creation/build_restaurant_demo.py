"""Create a separate first-service-demo layer without editing existing assets."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
dll = Path(sys.executable).parent / "Lib/site-packages/usd_exchange.libs"
if os.name == "nt" and dll.is_dir():
    _cookie = os.add_dll_directory(str(dll))
from robot.scene.restaurant_demo_build import build_demo_scene, build_demo_v2_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    result = (build_demo_scene if args.version == 1 else build_demo_v2_scene)(args.output, base_scene=args.base)
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
