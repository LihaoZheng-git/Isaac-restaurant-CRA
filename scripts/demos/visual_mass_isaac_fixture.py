"""Capture a stationary controlled Isaac visual-mass test inventory.

The rendered RGB and semantic masks are model inputs. Object dimensions and
authored masses are offline oracle labels; neither is live sensor evidence.
No robot or wheel command is created.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIRECTORY = Path(
    os.environ.get(
        "VISUAL_MASS_ISAAC_OUTPUT",
        ROOT / "outputs" / "visual_mass_isaac_fixture",
    )
)
os.environ.setdefault("WARP_CACHE_PATH", str(OUTPUT_DIRECTORY / "warp"))

from isaacsim import SimulationApp


HEADLESS = os.environ.get("ISAAC_SIM_HEADLESS", "0") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})


import isaacsim.core.experimental.utils.semantics as semantics_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.replicator.core as rep
import numpy as np
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

from robot.perception.visual_mass_collection import (
    VisualMassCapture,
    persist_visual_mass_capture,
)
from robot.perception.visual_mass_dataset import (
    validate_visual_mass_dataset,
    write_visual_mass_manifest,
)
from robot.perception.visual_mass_isaac_inventory import (
    ISAAC_MASS_TEST_INVENTORY,
    IsaacMassObject,
    mask_for_semantic_class,
)
from robot.scene.visual_mass_ground_truth import read_authored_mass_ground_truth


WIDTH = 320
HEIGHT = 320
CAMERA_VIEWS = (
    ("front-left", (1.05, -1.35, 1.05)),
    ("front-right", (-1.05, -1.35, 0.92)),
)
TARGET_PATH = "/World/Target"


def _material(stage, path: str, color: tuple[float, float, float]):
    result = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return result


def _bind(prim, material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _create_scene(item: IsaacMassObject):
    stage = stage_utils.create_new_stage(template="empty")
    stage_utils.set_stage_units(meters_per_unit=1.0, kilograms_per_unit=1.0)
    stage_utils.set_stage_up_axis("Z")
    target_material = _material(stage, "/World/Looks/Target", item.color)
    table_material = _material(stage, "/World/Looks/Table", (0.30, 0.22, 0.16))

    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.35))
    table.AddScaleOp().Set(Gf.Vec3f(0.7, 0.7, 0.05))
    _bind(table.GetPrim(), table_material)

    width, depth, height = item.dimensions_m
    if item.shape == "cylinder":
        target = UsdGeom.Cylinder.Define(stage, TARGET_PATH)
        target.CreateRadiusAttr(width / 2)
        target.CreateHeightAttr(height)
        prim = target.GetPrim()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, 0.40 + height / 2)
        )
        if abs(width - depth) > 1e-9:
            target.AddScaleOp().Set(Gf.Vec3f(1.0, depth / width, 1.0))
    else:
        target = UsdGeom.Cube.Define(stage, TARGET_PATH)
        target.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.40 + height / 2))
        target.AddScaleOp().Set(Gf.Vec3f(width / 2, depth / 2, height / 2))
        prim = target.GetPrim()
    _bind(prim, target_material)
    semantics_utils.add_labels(prim, labels=[item.semantic_class], taxonomy="class")
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(item.authored_mass_kg)

    light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    light.CreateIntensityAttr(1100.0)
    UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(42, 0, 28))
    return stage


def _capture_view(item: IsaacMassObject, camera_id: str, position, sample_id: str):
    camera = rep.create.camera(
        position=position,
        look_at=(0.0, 0.0, 0.53 + item.dimensions_m[2] / 2),
        focal_length=55.0,
        name=f"MassFixture-{sample_id}",
    )
    render_product = rep.create.render_product(
        camera,
        (WIDTH, HEIGHT),
        name=sample_id.replace("-", "_"),
    )
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    segmentation_annotator = rep.AnnotatorRegistry.get_annotator(
        "semantic_segmentation",
        init_params={"semanticTypes": ["class"], "colorize": False},
    )
    rgb_annotator.attach(render_product)
    segmentation_annotator.attach(render_product)
    try:
        rep.orchestrator.preview()
        for _ in range(6):
            simulation_app.update()
        rgb = np.asarray(())
        segmentation_data = {}
        for _ in range(8):
            rep.orchestrator.step(
                delta_time=0.0,
                rt_subframes=8,
                pause_timeline=False,
                wait_for_render=True,
            )
            simulation_app.update()
            rgb = np.asarray(rgb_annotator.get_data())
            segmentation_data = segmentation_annotator.get_data()
            if rgb.shape[:2] == (HEIGHT, WIDTH) and np.asarray(
                segmentation_data.get("data", ())
            ).size:
                break
        if rgb.shape[:2] != (HEIGHT, WIDTH):
            raise RuntimeError(f"unexpected RGB capture shape: {rgb.shape}")
        mask = mask_for_semantic_class(
            segmentation_data["data"],
            segmentation_data.get("info", {}).get("idToLabels", {}),
            item.semantic_class,
        )
        preview_path = OUTPUT_DIRECTORY / "previews" / f"{sample_id}.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb[:, :, :3].astype(np.uint8), mode="RGB").save(preview_path)
        return persist_visual_mass_capture(
            VisualMassCapture(
                sample_id=sample_id,
                object_id=item.object_id,
                semantic_class=item.semantic_class,
                scene_id=f"isaac-mass-fixture-{item.object_id}",
                split="test",
                mass_kg=item.authored_mass_kg,
                mass_source="isaac_authored_mass",
                dimensions_m=item.dimensions_m,
                camera_id=camera_id,
                simulation_seed=0,
                rgb=rgb[:, :, :3].astype(np.uint8),
                object_mask=mask,
            ),
            OUTPUT_DIRECTORY,
        )
    finally:
        rgb_annotator.detach(render_product)
        segmentation_annotator.detach(render_product)
        render_product.destroy()


def main() -> None:
    manifest_path = OUTPUT_DIRECTORY / "manifest.jsonl"
    if manifest_path.exists() or (OUTPUT_DIRECTORY / "rgb").exists():
        raise FileExistsError(
            "refusing to overwrite an existing fixture; set VISUAL_MASS_ISAAC_OUTPUT"
        )
    samples = []
    for item in ISAAC_MASS_TEST_INVENTORY:
        stage = _create_scene(item)
        truth = read_authored_mass_ground_truth(stage, TARGET_PATH)
        if not math.isclose(
            truth.mass_kg,
            item.authored_mass_kg,
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"authored mass mismatch for {item.object_id}")
        for view_index, (camera_id, position) in enumerate(CAMERA_VIEWS, 1):
            sample_id = f"{item.object_id}-view-{view_index:02d}"
            samples.append(_capture_view(item, camera_id, position, sample_id))
        print(f"captured={item.object_id}; mass_source={truth.source}", flush=True)
    write_visual_mass_manifest(samples, manifest_path)
    report = validate_visual_mass_dataset(samples, OUTPUT_DIRECTORY)
    if not report.valid or report.object_count != 10 or report.sample_count != 20:
        raise RuntimeError(f"invalid visual-mass fixture: {report}")
    print(f"manifest={manifest_path}")
    print(f"objects={report.object_count}; samples={report.sample_count}")
    print("geometry=isaac_authored_oracle; mass=isaac_authored_oracle")
    print("robot_motion=none; runtime_sensor_claim=none")
    print("PASS: stationary Isaac visual-mass fixture captured", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
