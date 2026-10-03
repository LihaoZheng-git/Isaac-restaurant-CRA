"""Stationary RGB plume-like baseline on the existing authored Isaac fixture.

No robot is created. Source region/context are explicit fixture priors; plume
masks are computed from RGB, never from renderer segmentation annotations.
"""

from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# This existing demo owns SimulationApp initialization and the approved proxy
# appearance. Reuse it without changing the restaurant stage or environment.
from scripts.demos.hot_soup_steam_isaac_demo import create_scene, simulation_app

import numpy as np
import omni.replicator.core as rep
import omni.timeline
from PIL import Image
from pxr import UsdGeom

from robot.perception.continuous_thermal_score import estimate_continuous_thermal_score
from robot.perception.hot_soup_isaac_variants import HOT_SOUP_ISAAC_VARIANTS
from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate
from robot.perception.plume_features import PlumeSourceObservation, extract_continuous_plume_features
from robot.perception.rgb_plume import StationaryRgbPlumeEstimator
from robot.perception.semantic_detection import PixelBoundingBox

OUTPUT = ROOT / "outputs" / "rgb_plume_isaac"
SOURCE_BOX = PixelBoundingBox(220, 290, 420, 398)


def prior(timestamp: float) -> ScalarEstimate:
    return ScalarEstimate(1., 1., 1., .5, "1", "controlled_open_container_roi_prior",
                          timestamp, EstimateStatus.PRIOR,
                          ("configured fixture context; generic model did not detect this cartoon container",))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stage = create_scene(HOT_SOUP_ISAAC_VARIANTS[0])
    plumes = [stage.GetPrimAtPath(f"/World/HotSoup/Plume_{index:02d}") for index in range(5)]
    camera = rep.create.camera(position=(2.3, -3.2, 1.75), look_at=(0., 0., .95), focal_length=38.)
    product = rep.create.render_product(camera, (640, 480), name="RgbPlumeDiagnostic")
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(product)

    def capture(visible_count: int):
        for index, prim in enumerate(plumes):
            imageable = UsdGeom.Imageable(prim)
            if index < visible_count:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        for _ in range(5):
            simulation_app.update()
        for _ in range(8):
            rep.orchestrator.step(delta_time=.1, rt_subframes=16, pause_timeline=False, wait_for_render=True)
            simulation_app.update()
            image = np.asarray(annotator.get_data())
            if image.shape == (480, 640, 4):
                return np.array(image[:, :, :3], dtype=np.uint8), float(omni.timeline.get_timeline_interface().get_current_time())
        raise RuntimeError("Isaac RGB capture is unavailable")

    try:
        rep.orchestrator.preview()
        reference, _ = capture(0)
        Image.fromarray(reference).save(OUTPUT / "reference.png")
        estimator = StationaryRgbPlumeEstimator(reference, source_box=SOURCE_BOX)
        sequences = {"low_cue": [0]*6, "intermittent_small": [1, 0, 1, 0, 1, 0], "persistent_large": [5]*6}
        reports = {}
        frame_id = 0
        for name, counts in sequences.items():
            observations = []
            masks = []
            times = []
            last_image = None
            for visible_count in counts:
                image, timestamp = capture(visible_count)
                result = estimator.estimate(image, source_frame_id=frame_id, timestamp_seconds=timestamp)
                frame_id += 1
                if result.frame is None:
                    raise RuntimeError(result.rejection_reasons)
                context = prior(timestamp)
                # Empty observations are accepted only in this controlled,
                # stationary-reference domain, checked against positive cases.
                relation = result.spatial_association if result.frame.instances else context
                observations.append(PlumeSourceObservation(result.frame, SOURCE_BOX, context,
                    relation, 0 if result.frame.instances else None,
                    negative_observation_validated=True, negative_observation_confidence=.55))
                mask = result.frame.instances[0].mask if result.frame.instances else np.zeros(image.shape[:2], dtype=bool)
                masks.append(mask)
                times.append(timestamp)
                last_image = image
            features = extract_continuous_plume_features(tuple(observations), timestamp_seconds=times[-1])
            score = estimate_continuous_thermal_score(features, timestamp_seconds=times[-1])
            estimate = score.thermal.heat_source_likelihood
            reports[name] = {
                "thermal_visual_score": estimate.value, "lower_bound": estimate.lower_bound,
                "upper_bound": estimate.upper_bound, "confidence": estimate.confidence,
                "status": estimate.status.value, "source": estimate.source,
                "positive_frames": features.positive_frame_count,
                "invoked_frames": features.invoked_frame_count,
                "relative_mask_area": features.relative_mask_area.value,
                "vertical_extent": features.vertical_extent.value,
                "persistence": features.persistence.value,
                "rejection_reasons": score.rejection_reasons,
            }
            np.savez_compressed(OUTPUT / f"{name}.npz", masks=np.stack(masks), timestamps_seconds=np.asarray(times))
            overlay = np.array(last_image, copy=True)
            union = np.any(masks, axis=0)
            overlay[union] = (.45*overlay[union] + .55*np.asarray((30, 240, 70))).astype(np.uint8)
            Image.fromarray(overlay).save(OUTPUT / f"{name}-overlay.png")
        values = [reports[name]["thermal_visual_score"] for name in sequences]
        passed = all(value is not None for value in values) and values[0] < values[1] < values[2]
        passed = passed and reports["low_cue"]["positive_frames"] == 0
        payload = {"passed": passed, "cases": reports, "config": asdict(estimator.config),
                   "robot_motion": "no robot created", "plume_mask_provenance": "inferred_from_RGB",
                   "source_context_provenance": "prior", "source_box": asdict(SOURCE_BOX),
                   "temperature": "unavailable", "steam_identity": "unavailable",
                   "negative_validation_scope": "this fixed-camera rendered fixture only",
                   "limitations": ["clean reference and supplied source region required",
                       "not general steam/smoke recognition", "grey foreground can confound the method",
                       "cartoon USD proxy; no fluid or heat simulation", "no live four-factor sensor acceptance"]}
        (OUTPUT / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
        if not passed:
            raise RuntimeError("controlled continuous RGB plume ordering did not pass")
    finally:
        annotator.detach(product)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
