"""Create a separate angular or rounded service-bin scene for stationary perception."""

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
from robot.scene.restaurant_container_bench import build_container_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", choices=("angular", "rounded"), required=True)
    parser.add_argument("--props-only", action="store_true", help="isolate the stationary prop test by disabling the background server in this new layer only")
    args = parser.parse_args()
    result = build_container_scene(args.output, base_scene=args.base, shape=args.shape, props_only=args.props_only)
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
