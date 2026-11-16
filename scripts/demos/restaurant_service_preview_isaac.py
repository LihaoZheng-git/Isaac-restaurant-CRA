"""Render the actual dressed restaurant without advancing physics or wheels."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom
from robot.scene.controlled_semantic_obstacle import read_world_bounds
from robot.scene.restaurant_service_build import SERVICE_ROOT, service_props
from robot.scene.restaurant_server import SERVER_PRIM_PATH
from robot.scene.restaurant_demo_build import DEMO_ROOT, demo_prop_records, payload_geometry_check


def main():
    sources = {ROOT/"scenes/generated/restaurant_base.usda", ROOT/"scenes/generated/restaurant_robot.usda"}
    pending = [args.scene.resolve()]
    while pending:
        path = pending.pop()
        if path in sources:
            continue
        sources.add(path)
        layer = Sdf.Layer.FindOrOpen(str(path))
        pending.extend(Path(Sdf.ComputeAssetPathRelativeToLayer(layer,p)).resolve() for p in layer.subLayerPaths)
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if not omni.usd.get_context().open_stage(str(args.scene.resolve())):
        raise RuntimeError("scene did not open")
    for _ in range(100):
        app.update()
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    stage = omni.usd.get_context().get_stage()
    root_before = stage.GetRootLayer().ExportToString()
    bounds_before = {p.name:read_world_bounds(stage, f"{SERVICE_ROOT}/{p.name}") for p in service_props()}
    has_server = stage.GetPrimAtPath(SERVER_PRIM_PATH).IsValid()
    if has_server:
        bounds_before["Server"] = read_world_bounds(stage,SERVER_PRIM_PATH)
    views = (
        ("overview", (37, 1, 16), (26, 9, 0), 24),
        ("food-table", (29.8, 5.5, 1.75), (31.25, 7.4, .86), 38),
        ("service-counter", (26.2, 4.4, 2.0), (24.5, 6.05, 1.0), 42),
        ("second-table", (29.3, 10.4, 1.7), (31.2, 12.25, .9), 38),
        ("front-table", (14.4, 3.6, 1.8), (13.5, 2.55, .8), 38),
    )
    if has_server:
        views += (("server",(26.3,7.3,1.6),(29.8,8.9,.9),26),
                  ("server-aisle",(25.5,5.8,2.2),(29.4,8.9,.8),24))
    demo_before = demo_prop_records(stage)
    if stage.GetPrimAtPath(DEMO_ROOT):
        # Interior views stay below the restaurant ceiling. No hidden geometry.
        views = (
            ("service-aisle", (25.9, 6.7, 3.2), (29.2, 8.3, .7), 18),
            ("clearing-station", (28.6, 4.3, 1.8), (30.7, 5.5, .65), 38),
            ("rounded-bin", (29.3, 6.1, 1.1), (30.65, 6.05, .35), 42),
            ("utensil-comparison", (29.7, 5.45, 1.35), (30.7, 5.45, .95), 24),
            ("dining-table", (28.7, 9.7, 1.7), (31.0, 9.8, .65), 32),
        )
        body = stage.GetPrimAtPath("/World/Robot/chassis_link")
        origin = UsdGeom.XformCache().GetLocalToWorldTransform(body).ExtractTranslation()
        views += (("serving-robot", tuple(origin+Gf.Vec3d(-1.4, -1.2, 1.2)),
                   tuple(origin+Gf.Vec3d(-.25, 0, .25)), 32),)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for name,eye,target,focal in views:
            camera = rep.create.camera(position=eye, look_at=target, focal_length=focal, name=f"Service_{name.replace('-', '_')}")
            product = rep.create.render_product(camera, (960,640), name=f"Service_{name.replace('-', '_')}")
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb.attach(product)
            try:
                for _ in range(10):
                    app.update()
                frame = None
                for _ in range(4):
                    rep.orchestrator.step(delta_time=0.0, rt_subframes=8, pause_timeline=True, wait_for_render=True)
                    app.update()
                    data = np.asarray(rgb.get_data())
                    if data.shape[:2] == (640,960):
                        frame = data[:,:,:3].astype(np.uint8)
                        break
                if frame is None:
                    raise RuntimeError(f"missing render for {name}")
                Image.fromarray(frame).save(args.output / f"{name}.png")
                print(f"saved {name}", flush=True)
            finally:
                rgb.detach(product)
                product.destroy()
    report = {"source_files_unchanged":all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in sources),
              "source_sha256":hashes,"root_layer_unchanged":stage.GetRootLayer().ExportToString()==root_before,
              "prop_bounds_unchanged":all(read_world_bounds(stage,f"{SERVICE_ROOT}/{p.name}")==bounds_before[p.name] for p in service_props())
                  and (not has_server or read_world_bounds(stage,SERVER_PRIM_PATH)==bounds_before["Server"]),
              "demo_prop_bounds_unchanged": demo_prop_records(stage) == demo_before,
              "serving_payload": payload_geometry_check(stage),
              "in_memory_renderer_edits_allowed":True,
              "timeline_seconds":timeline.get_current_time(),"wheel_commands":0,
              "humans_added":int(has_server),"actor_motion_enabled":False,"scope":"actual Isaac RGB preview; no CRA or navigation acceptance"}
    report["passed"] = report["source_files_unchanged"] and report["prop_bounds_unchanged"] and report["demo_prop_bounds_unchanged"] and report["timeline_seconds"] == 0
    (args.output/"preview-report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if not report["root_layer_unchanged"]:
        import difflib
        (args.output/"in-memory-renderer-root.diff").write_text("\n".join(difflib.unified_diff(
            root_before.splitlines(),stage.GetRootLayer().ExportToString().splitlines())),encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError("preview scene/time invariant failed")


try:
    main()
finally:
    app.close()
