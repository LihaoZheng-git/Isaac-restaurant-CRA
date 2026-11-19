"""Prepare the restaurant scene using the bundled assets and NVIDIA's asset packs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.error import URLError
from zipfile import BadZipFile

if __package__:
    from . import vendor
else:  # Also support the original "python experiment/scene/prepare.py" command.
    import vendor


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
LAYERS = HERE / "layers"
ASSETS = HERE.parents[1] / "assets"
PROJECT_LIVE = ASSETS / "restaurant" / "live"
DEFAULT_CACHE = Path.home() / ".cache" / "isaac-restaurant-cra" / "v2"
NOTICES = (
    "README.md",
    "restaurant_live_v1/assets/licenses/plate-trolley.txt",
    "restaurant_live_v1/assets/licenses/pink-trolley.txt",
    "restaurant_live_v5/characters/rocketbox/LICENSE.md",
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def install_live_assets(cache, expected):
    live = cache / "live"
    bundled = {name: digest for name, digest in expected.items() if not name.startswith("restaurant_live_v3/")}
    bundled.update({name: sha256(PROJECT_LIVE / name) for name in NOTICES})
    for name, fingerprint in bundled.items():
        source = PROJECT_LIVE / name
        if not source.is_file() or sha256(source) != fingerprint:
            raise ValueError(f"Missing or modified bundled asset: {source}. Restore it from Git.")
        target = live / name
        if target.is_file() and sha256(target) == fingerprint:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as temp:
            copy = Path(temp) / target.name
            shutil.copyfile(source, copy)
            copy.replace(target)
    vendor.adult_characters(live, expected)
    return live


def prepare(output, restaurant, live):
    output = output.expanduser().resolve()
    locations = {"restaurant": restaurant.expanduser().resolve(), "live": live.expanduser().resolve(), "assets": ASSETS}
    for path in locations.values():
        if "@" in path.as_posix():
            raise ValueError(f"USD asset paths must not contain the @ delimiter: {path}")
    inputs = load("inputs.json")
    sources = {}
    for group, root in {**locations, "layers": LAYERS}.items():
        for relative, expected in inputs[group].items():
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"Missing {group} asset: {path}")
            if sha256(path) != expected:
                raise ValueError(f"Asset differs from the recorded v2 input: {path}")
            if group != "layers":
                sources[str(path)] = expected

    # The manifests name their final locations. Publish the fully written bundle
    # in one rename so a failed preparation leaves no partial runnable scene.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / "prepared"
        generated = bundle / "scenes/generated"
        for relative in inputs["layers"]:
            layer = (LAYERS / relative).read_bytes()
            for group, root in locations.items():
                layer = layer.replace(("@{" + group + "}/").encode(), ("@" + root.as_posix() + "/").encode())
            target = generated / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(layer)
            sources[str(output / target.relative_to(bundle))] = sha256(target)

        monitor = bundle / "outputs/restaurant-live-cra/authored-monitor"
        survey = load("geometry-survey.json")
        survey["source_sha256"] = {
            str(output / "scenes/generated" / name): sha256(generated / name)
            for name in survey["source_sha256"]
        }
        write_json(monitor / "geometry-survey.json", survey)
        mask = monitor / "authored-height-column-mask.png"
        mask.write_bytes((DATA / mask.name).read_bytes())
        if sha256(mask) != survey["mask_sha256"]:
            raise ValueError("The v2 authored survey mask has changed")

        cast = load("cast.json")
        for actor in cast["actors"].values():
            if "asset_path" in actor:
                actor["asset_path"] = (locations["live"] / "restaurant_live_v3" / actor["asset_path"]).as_posix()
        cast_path = generated / "restaurant_live_v3/cast.json"
        write_json(cast_path, cast)
        write_json(bundle / "asset-locations.json", {k: str(path) for k, path in locations.items()})
        for path in (monitor / "geometry-survey.json", mask, cast_path, bundle / "asset-locations.json"):
            sources[str(output / path.relative_to(bundle))] = sha256(path)
        layout = load("service-layout.json")
        manifest = (bundle / layout["scene"]).with_suffix(".json")
        for name in ("scene", "base_scene", "additions", "bar_cast_manifest"):
            layout[name] = str(output / layout[name])
        layout["source_sha256"] = sources
        write_json(manifest, layout)
        try:
            bundle.rename(output)
        except OSError:
            if not output.is_dir():
                raise
            for item in bundle.rglob("*"):
                if item.is_file():
                    existing = output / item.relative_to(bundle)
                    if not existing.is_file() or sha256(existing) != sha256(item):
                        raise ValueError(
                            f"Existing scene at {output} differs. Choose a new --output directory."
                        ) from None
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="Where to store NVIDIA downloads and runtime assets")
    parser.add_argument("--restaurant-pack", type=Path, help="Reuse an extracted NVIDIA Restaurant Demo Assets Pack")
    parser.add_argument("--live-generated", type=Path, help="Reuse an existing complete restaurant_live_v1/v3/v5 directory")
    parser.add_argument("--output", default=HERE / "prepared", type=Path, help="Prepared scene directory (default: experiment/scene/prepared)")
    args = parser.parse_args()
    try:
        cache = args.cache_dir.expanduser()
        restaurant = args.restaurant_pack or vendor.restaurant_pack(cache)
        live = args.live_generated or install_live_assets(cache, load("inputs.json")["live"])
        output = prepare(args.output, restaurant, live)
    except (OSError, ValueError, URLError, BadZipFile) as error:
        parser.exit(1, f"Scene setup failed: {error}\n")
    print(f"Prepared scene: {output / load('service-layout.json')['scene']}")


if __name__ == "__main__":
    main()
