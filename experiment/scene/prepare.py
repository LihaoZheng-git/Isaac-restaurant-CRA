"""Prepare the v2 scene from the project's USD layers and local asset packs."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
LAYERS = HERE / "layers"
ASSETS = HERE.parents[1] / "assets"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def prepare(output, restaurant, live):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    locations = {"restaurant": restaurant.resolve(), "live": live.resolve(), "assets": ASSETS}
    for path in locations.values():
        if "@" in path.as_posix():
            raise ValueError(f"USD asset paths must not contain the @ delimiter: {path}")
    inputs = load("inputs.json")
    sources = {}
    for group, root in {**locations, "layers": LAYERS}.items():
        for relative, expected in inputs[group].items():
            path = root / relative
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
        bundle.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restaurant-pack", required=True, type=Path, help="Extracted AECO restaurant pack root")
    parser.add_argument("--live-generated", required=True, type=Path, help="Folder with restaurant_live_v1/v3/v5")
    parser.add_argument("--output", required=True, type=Path, help="New prepared-scene directory")
    args = parser.parse_args()
    print(prepare(args.output, args.restaurant_pack, args.live_generated))


if __name__ == "__main__":
    main()
