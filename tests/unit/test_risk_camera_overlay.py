"""Presentation must preserve accepted ownership, score provenance and source pixels."""

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from robot.evaluation.risk_camera_overlay import (
    capture_risk_camera_frame, overlay_rgb, render_risk_panel, risk_color_rgb,
)
from robot.perception.semantic_detection import (
    PixelBoundingBox, SemanticDetection, SemanticDetectionFrame,
)
from test_live_object_perception import packet
from test_object_risk_map import mapped_worker


def fixture(phi=.2, applied=.8, track_id=7):
    source = packet()
    mask = np.zeros(source.rgb.shape[:2], bool)
    mask[20:80, 20:80] = True
    detection = SemanticDetection(0, "unknown", .7, PixelBoundingBox(20, 20, 80, 80), mask)
    frame = SemanticDetectionFrame(source.camera_frame_id, source.camera_timestamp_seconds,
                                    100, 100, (detection,), "unit_fixture", 0.)
    attrs = NS(track_id=track_id)
    obj = NS(attributes=attrs, policy=NS(risk=phi), diagnostics=dict(
        track_id=track_id, accepted_instance=True, selected_return_count=4,
        factor_scores={name: dict(mode="configured_prior" if name == "thermal" else "evidence_supported")
                       for name in ("thermal", "contact", "potential", "kinetic")}))
    raw = dict(attributes=attrs, accepted=True, selected=np.ones(4, bool), frame=frame)
    return source, obj, raw, {track_id: NS(risk=applied)}


def capture(source, obj, raw, policies):
    return capture_risk_camera_frame(source, [obj], [raw], policies)


def test_only_current_verified_mask_preserves_individual_and_applied_scores():
    source, obj, raw, policies = fixture()
    frame = capture(source, obj, raw, policies)
    item, = frame.masks
    assert item.track_id == 7 and item.phi == .2 and item.applied_phi == .8
    assert item.basis == "evidence + priors*"
    assert item.diagnostic()["factor_modes"]["thermal"] == "configured_prior"
    assert not frame.rgb.flags.writeable and not item.mask.flags.writeable
    assert not np.shares_memory(frame.rgb, source.rgb)
    assert not np.shares_memory(item.mask, raw["frame"].detections[0].mask)
    np.testing.assert_array_equal(frame.rgb, source.rgb)
    assert frame.camera_timestamp_seconds == source.camera_timestamp_seconds
    assert frame.sensor_timestamp_seconds == source.sensor_timestamp_seconds


def test_human_contact_prior_does_not_masquerade_as_factor_evidence():
    source, obj, raw, policies = fixture()
    for name, value in obj.diagnostics["factor_scores"].items():
        value["mode"] = "configured_human_body_contact_prior" if name == "contact" else "configured_prior"
    item, = capture(source, obj, raw, policies).masks
    assert item.basis == "factor priors"


@pytest.mark.parametrize("fault", ["id", "frame", "time", "support", "count", "shape", "phi", "applied"])
def test_inconsistent_presentation_records_are_withheld_without_changing_policy(fault):
    source, obj, raw, policies = fixture()
    if fault == "id":
        raw["attributes"] = NS(track_id=8)
    elif fault == "frame":
        raw["frame"] = replace(raw["frame"], source_frame_id=2)
    elif fault == "time":
        raw["frame"] = replace(raw["frame"], source_timestamp_seconds=1.2)
    elif fault == "support":
        raw["selected"][:] = False
    elif fault == "count":
        obj.diagnostics["selected_return_count"] = 3
    elif fault == "shape":
        source = replace(source, rgb=np.zeros((120, 100, 3), np.uint8))
    elif fault == "phi":
        obj.policy.risk = float("nan")
    else:
        policies[7].risk = 1.1
    before = dict(obj.diagnostics)
    frame = capture(source, obj, raw, policies)
    assert not frame.masks and len(frame.rejected) == 1
    assert obj.diagnostics == before


@pytest.mark.parametrize("rejected_side", ["object", "assessment"])
def test_retained_or_raw_masks_do_not_become_accepted_masks(rejected_side):
    source, obj, raw, policies = fixture()
    if rejected_side == "object":
        obj.diagnostics["accepted_instance"] = False
    else:
        raw["accepted"] = False
    assert not capture(source, obj, raw, policies).masks


def test_no_masks_never_promotes_geometric_baseline_to_a_visual_instance():
    source = packet()
    enabled, disabled = mapped_worker(), mapped_worker()
    enabled.capture_risk_masks = True
    a, b = enabled(source), disabled(source)
    assert a.risk_camera_frame is not None and not a.risk_camera_frame.masks
    assert b.risk_camera_frame is None
    assert {k: v for k, v in a.diagnostics.items() if k != "stage_wall_seconds"} == {
        k: v for k, v in b.diagnostics.items() if k != "stage_wall_seconds"}
    assert a.policy.risk == b.policy.risk
    np.testing.assert_array_equal(a.field.tracking_result.point_track_ids,
                                  b.field.tracking_result.point_track_ids)


def test_real_all_track_adapter_captures_accepted_mask_without_changing_field():
    points = np.array([(1+x, y, z) for x in (-.08, 0, .08)
                       for y in (-.08, 0, .08) for z in (.02, .18, .34)])
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
    source = replace(packet(), obstacle_xyz=points, scene_xyz=points,
                     camera_from_scan_robot=transform)
    mask = np.zeros((100, 100), bool)
    mask[10:90, 10:90] = True
    detection = SemanticDetection(0, "unknown", .5, PixelBoundingBox(10, 10, 90, 90), mask)
    results, calls = [], []
    for enabled in (False, True):
        worker = mapped_worker()
        worker.capture_risk_masks = enabled
        from robot.perception.physical_attributes import MassAttributes
        from test_object_risk_map import scalar
        worker.mass_estimator = NS(estimate_instance=lambda *a, **kw:
                                  MassAttributes(scalar(2, "kg")))
        def propose(*args, **kwargs):
            calls.append(enabled)
            return NS(detection=detection, model_quality=.95, rejection_reasons=(),
                      candidate_diagnostics=(), prompt_xy=(50, 40),
                      inference_wall_seconds=0., image_encoder_wall_seconds=0.)
        worker.segmenter = NS(set_image=lambda rgb: None, propose=propose)
        results.append(worker(source))
    a, b = results
    assert calls == [False, True]  # No extra model call to make the presentation.
    assert a.risk_camera_frame is None
    captured, = b.risk_camera_frame.masks
    assert captured.phi == b.objects[0].policy.risk
    assert captured.track_id == b.objects[0].attributes.track_id
    assert a.objects[0].diagnostics == b.objects[0].diagnostics
    for ga, gb in zip(a.field.pipeline.configuration_space.occupancy_grids,
                      b.field.pipeline.configuration_space.occupancy_grids):
        np.testing.assert_array_equal(ga.occupied_mask, gb.occupied_mask)
        np.testing.assert_array_equal(ga.cell_risks, gb.cell_risks)


def test_fixed_scale_overlap_and_no_input_pixel_modification():
    source, obj, raw, policies = fixture()
    frame = capture(source, obj, raw, policies)
    a = frame.masks[0]
    b = replace(a, track_id=8, phi=.9)
    original = source.rgb.copy()
    first = overlay_rgb(replace(frame, masks=(a, b)))
    second = overlay_rgb(replace(frame, masks=(b, a)))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(source.rgb, original)
    np.testing.assert_array_equal(first[50, 50], np.rint(.38*np.array(risk_color_rgb(.9))))
    np.testing.assert_array_equal(first[95, 95], original[95, 95])
    assert risk_color_rgb(0) == (35, 195, 85)
    assert risk_color_rgb(.5) == (245, 220, 45)
    assert risk_color_rgb(.75) == (245, 135, 35)
    assert risk_color_rgb(1) == (225, 45, 45)
    assert np.linalg.norm(np.subtract(risk_color_rgb(.5001), risk_color_rgb(.4999))) <= 2


def test_held_assessment_uses_original_rgb_and_shows_age_and_invalid_time():
    frame = capture(*fixture())
    fresh, f = render_risk_panel(frame, 1.1)
    old, o = render_risk_panel(frame, 3.1)
    assert o["source_age_seconds"] == 2 and f["source_age_seconds"] == 0
    np.testing.assert_array_equal(fresh[58:458], old[58:458])
    empty, invalid = render_risk_panel(frame, 1.)
    assert invalid["status"] == "invalid_presentation_time" and not invalid["masks"]
    assert empty.shape == (600, 640, 3)
    assert render_risk_panel(None, 2)[1]["status"] == "awaiting_assessment"


def test_new_assessment_without_masks_does_not_keep_previous_colored_objects(tmp_path):
    from robot.scene.service_demo_video import ServiceDemoVideo
    recorder = ServiceDemoVideo.__new__(ServiceDemoVideo)
    recorder.output, recorder.risk_writer, recorder.risk_rejections = tmp_path, object(), []
    (tmp_path/"risk-camera-sources").mkdir()
    frame = capture(*fixture())
    result = NS(risk_camera_frame=frame, field=NS(sensor_frame_id=frame.sensor_frame_id,
                sensor_timestamp_seconds=frame.sensor_timestamp_seconds))
    recorder.accept_risk_assessment(result)
    assert len(recorder.risk_frame.masks) == 1
    result.risk_camera_frame = replace(frame, masks=())
    recorder.accept_risk_assessment(result)
    assert not recorder.risk_frame.masks
    result.risk_camera_frame = frame
    result.field.sensor_frame_id += 1
    recorder.accept_risk_assessment(result)
    assert recorder.risk_frame is None and recorder.risk_rejections


def test_real_stationary_cli_accepts_presentation_flag_before_kit(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    recipe = json.loads((root/"configs/restaurant_risk_camera_v1.json").read_text())
    command = recipe["commands"]["camera-preview"]
    source = root/command[0]
    prefix = source.read_text().split("args.output.mkdir", 1)[0]
    monkeypatch.setattr(sys, "argv", [*command, "--output", str(tmp_path/"native")])
    monkeypatch.setattr(sys, "path", list(sys.path))
    scope = {"__file__": str(source), "__name__": "cli_validation_test"}
    exec(compile(prefix, str(source), "exec"), scope)
    assert scope["args"].presentation_risk_camera and not scope["args"].motion
    assert not (tmp_path/"native").exists()
