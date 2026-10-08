"""Stationary object-plume policy on the unchanged rendered plume fixture.

RGB supplies plume features. Source regions are explicitly configured PRIORS;
this is a component diagnostic, not a successful general object detector run.
No robot is created. No authored segmentation mask supplies a thermal score.
"""

from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.demos.rgb_plume_isaac_diagnostic import (
    HOT_SOUP_ISAAC_VARIANTS,
    SOURCE_BOX,
    create_scene,
    simulation_app,
)

import numpy as np
import omni.replicator.core as rep
import omni.timeline
from PIL import Image
from pxr import UsdGeom

from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.object_plume_evidence import (
    ObjectPlumeObservation,
    estimate_object_plume_evidence,
)
from robot.perception.physical_attributes import EstimateStatus
from robot.perception.semantic_detection import (
    PixelBoundingBox,
    SemanticDetection,
    SemanticDetectionFrame,
)
from robot.perception.specialist_track_evidence import SpecialistAssociationFrame

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output", type=Path, default=ROOT / "outputs/object_plume/isaac_controls"
)
OUTPUT = parser.parse_args().output


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stage = create_scene(HOT_SOUP_ISAAC_VARIANTS[0])
    plumes = [
        stage.GetPrimAtPath(f"/World/HotSoup/Plume_{index:02d}") for index in range(5)
    ]
    camera = rep.create.camera(
        position=(2.3, -3.2, 1.75), look_at=(0.0, 0.0, 0.95), focal_length=38.0
    )
    product = rep.create.render_product(camera, (640, 480), name="ObjectPlumePolicy")
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(product)

    def capture(count):
        for index, prim in enumerate(plumes):
            imageable = UsdGeom.Imageable(prim)
            imageable.MakeVisible() if index < count else imageable.MakeInvisible()
        for _ in range(5):
            simulation_app.update()
        for _ in range(8):
            rep.orchestrator.step(
                delta_time=0.1,
                rt_subframes=16,
                pause_timeline=False,
                wait_for_render=True,
            )
            simulation_app.update()
            data = np.asarray(annotator.get_data())
            if data.shape == (480, 640, 4):
                return np.array(data[:, :, :3], dtype=np.uint8), float(
                    omni.timeline.get_timeline_interface().get_current_time()
                )
        raise RuntimeError("stationary Isaac RGB capture unavailable")

    frame_id = 0

    def observation(rgb, timestamp, boxes=(SOURCE_BOX,)):
        nonlocal frame_id
        frame_id += 1
        detections, links = [], []
        for index, box in enumerate(boxes):
            mask = np.zeros(rgb.shape[:2], bool)
            mask[int(box.y_min) : int(box.y_max), int(box.x_min) : int(box.x_max)] = (
                True
            )
            detections.append(
                SemanticDetection(index, "configured_object_region", 0.5, box, mask)
            )
            links.append(
                TrackSemanticAssociation(
                    index + 1, "matched", 0, index, 0, 1.0, 1.0, True, box.center_xy
                )
            )
        return ObjectPlumeObservation(
            rgb,
            SemanticDetectionFrame(
                frame_id,
                timestamp,
                480,
                640,
                tuple(detections),
                "configured region PRIOR",
                0.0,
            ),
            SpecialistAssociationFrame(frame_id, timestamp, 480, 640, tuple(links)),
            timestamp,
            True,
            EstimateStatus.PRIOR,
        )

    try:
        rep.orchestrator.preview()
        reference, reference_time = capture(0)
        Image.fromarray(reference).save(OUTPUT / "reference.png")
        reports, saved = {}, {}
        for name, counts in {
            "no_plume": [0] * 3,
            "intermittent": [1, 0, 1],
            "dense": [5] * 3,
        }.items():
            frames = tuple(observation(*capture(count)) for count in counts)
            saved[name] = frames
            result = estimate_object_plume_evidence(
                reference,
                reference_timestamp_seconds=reference_time,
                observations=frames,
                timestamp_seconds=frames[-1].detections.source_timestamp_seconds,
                clean_reference_validated=True,
                negative_validation_scope="controlled stationary RGB fixture with positive/negative controls",
            )[1]
            reports[name] = asdict(result)
            np.savez_compressed(
                OUTPUT / f"{name}.npz",
                rgb=np.stack([item.rgb for item in frames]),
                timestamps=[
                    item.detections.source_timestamp_seconds for item in frames
                ],
            )
            Image.fromarray(frames[-1].rgb).save(OUTPUT / f"{name}.png")
        # Same actual RGB plume, but a distant source region: no source-class
        # assumption may transfer the plume to this unrelated object region.
        remote = tuple(
            observation(
                item.rgb,
                item.detections.source_timestamp_seconds,
                (SOURCE_BOX, PixelBoundingBox(20, 290, 110, 398)),
            )
            for item in saved["dense"]
        )
        for name, frames, clean, selected_track in (
            ("unrelated", remote, True, 2),
            ("unvalidated_reference", saved["no_plume"], False, 1),
            (
                "moving_camera",
                tuple(
                    replace(item, camera_stationary=False) for item in saved["no_plume"]
                ),
                True,
                1,
            ),
        ):
            result = estimate_object_plume_evidence(
                reference,
                reference_timestamp_seconds=reference_time,
                observations=frames,
                timestamp_seconds=frames[-1].detections.source_timestamp_seconds,
                clean_reference_validated=clean,
                negative_validation_scope="controlled fixture",
            )[selected_track]
            reports[name] = asdict(result)

        def heat(name):
            return reports[name]["thermal"]["heat_source_likelihood"]["value"]

        passed = (
            heat("no_plume") is not None
            and heat("no_plume") < heat("intermittent") < heat("dense")
            and heat("unrelated") is not None
            and heat("unrelated") < heat("dense")
            and heat("unvalidated_reference") is None
            and heat("moving_camera") is None
        )
        report = {
            "passed": passed,
            "cases": reports,
            "object_region_provenance": "prior",
            "plume_provenance": "inferred from actual RGB",
            "temperature": "unavailable",
            "robot_motion": "no robot created",
            "appearance": "unchanged approved USD plume proxy",
            "limitations": [
                "fixed camera and validated clean reference required",
                "source regions are configured priors",
                "grey foreground confound; no aerosol identity or temperature",
                "not a general real-camera negative validation",
            ],
        }
        (OUTPUT / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {"passed": passed, "scores": {name: heat(name) for name in reports}},
                indent=2,
            ),
            flush=True,
        )
        if not passed:
            raise RuntimeError("object plume stationary controls did not pass")
    finally:
        annotator.detach(product)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
