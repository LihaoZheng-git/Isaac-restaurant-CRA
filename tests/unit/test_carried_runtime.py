from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from test_object_risk_map import mapped_worker
from test_live_object_perception import packet
from robot.perception.carried_object import CarriedObjectConfig
from robot.perception.human_mass_prior import (
    HumanMassPriorConfig,
    HUMAN_MASS_PRIOR_SOURCE,
)
from robot.perception.human_pose import HumanPose, HumanPoseFrame
from robot.perception.physical_attributes import (
    MassAttributes,
    ScalarEstimate,
    EstimateStatus,
)
from robot.perception.sam_instance_segmentation import SamInstanceResult
from robot.perception.semantic_detection import (
    SemanticDetection,
    SemanticDetectionFrame,
    PixelBoundingBox,
)
from robot.perception.visual_mass_runtime import VISUAL_MASS_RUNTIME_SOURCE


def setup(monkeypatch):
    import robot.evaluation.carried_object_runtime as bridge

    runtime = mapped_worker()
    runtime.grid_config = replace(runtime.grid_config, resolution=0.1)
    runtime.carried_object_config = CarriedObjectConfig()
    runtime.human_mass_prior_config = HumanMassPriorConfig("controlled unit adult")
    runtime.persistent_properties.config = replace(
        runtime.persistent_properties.config,
        retain_observed_properties=False,
        allow_instance_bounds_prior=False,
        minimum_visual_reassessment_interval_seconds=0.2,
    )
    human_mask = np.zeros((100, 100), bool)
    human_mask[10:96, 40:60] = True
    object_mask = np.zeros((100, 100), bool)
    object_mask[45:65, 61:81] = True
    hd = SemanticDetection(
        0, "person", 0.9, PixelBoundingBox(40, 10, 60, 96), human_mask
    )
    od = SemanticDetection(
        0, "unknown", 0.75, PixelBoundingBox(61, 45, 81, 65), object_mask
    )
    keypoints = np.zeros((17, 3))
    keypoints[9] = [62, 55, 0.9]
    pose = HumanPose((40, 10, 60, 96), 0.9, keypoints)
    runtime.person_detector = SimpleNamespace(
        detect=lambda rgb, **kw: SemanticDetectionFrame(
            kw["source_frame_id"],
            kw["source_timestamp_seconds"],
            100,
            100,
            (hd,),
            "controlled_person",
            0.0,
        )
    )
    runtime.pose_detector = SimpleNamespace(
        detect=lambda rgb, **kw: HumanPoseFrame(
            kw["source_frame_id"], kw["source_timestamp_seconds"], (100, 100), (pose,)
        )
    )
    runtime.segmenter = SimpleNamespace(
        set_image=lambda rgb: None,
        propose=lambda *a, **kw: SamInstanceResult(
            None, ("unit generic proposal unavailable",)
        ),
    )
    runtime.mass_estimator = SimpleNamespace(
        estimate_instance=lambda *a, **kw: MassAttributes(
            ScalarEstimate(
                2.0,
                1.0,
                3.0,
                0.5,
                "kg",
                VISUAL_MASS_RUNTIME_SOURCE,
                kw["timestamp_seconds"],
                EstimateStatus.INFERRED,
            )
        )
    )

    def parent(p, *args):
        selected = np.zeros(len(p.obstacle_xyz), bool)
        selected[:16] = True
        return {
            "selected": selected,
            "proposal": SamInstanceResult(
                replace(hd, semantic_class="unknown"), (), model_quality=0.9
            ),
        }, {}

    def child(p, *args, **kwargs):
        selected = np.zeros(len(p.obstacle_xyz), bool)
        selected[16:] = True
        return {
            "selected": selected,
            "proposal": SamInstanceResult(od, (), model_quality=0.9),
            "hand_robot_xyz": p.obstacle_xyz[16],
            "wrist_confidence": 0.9,
        }, []

    monkeypatch.setattr(bridge, "propose_observed_person", parent)
    monkeypatch.setattr(bridge, "propose_carried_object", child)
    return runtime


def observed_packet(frame, t):
    # Controlled numerical geometry, not an Isaac/camera validation result.
    body = np.array(
        [
            [x, y, z]
            for x in (2.0, 2.04)
            for y in (-0.08, 0.08)
            for z in (0.05, 0.6, 1.1, 1.6)
        ]
    )
    obj = np.array(
        [[x, y, z] for x in (2.0, 2.04) for y in (-0.36, -0.34) for z in (0.85, 0.9)]
    )
    xyz = np.vstack((body, obj)) + [0.5 * (t - 1), 0, 0]
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
    transform[1, 3] = -1.0
    return replace(
        packet(frame, t),
        obstacle_xyz=xyz,
        scene_xyz=xyz,
        camera_from_scan_robot=transform,
    )


def test_runtime_keeps_member_mass_and_scores_separate_but_applies_max_to_both_geometries(
    monkeypatch,
):
    runtime = setup(monkeypatch)
    for frame, t in enumerate((1.0, 1.2, 1.4), 1):
        result = runtime(observed_packet(frame, t))
    detail = result.diagnostics["carried_object_groups"]
    assert len(detail["confirmed_groups"]) == 1
    group = detail["confirmed_groups"][0]
    members = [o for o in result.objects if o.diagnostics.get("carried_member")]
    assert len(members) == 2
    h = next(o for o in members if o.diagnostics["carried_member"] == "human")
    obj = next(o for o in members if o.diagnostics["carried_member"] == "object")
    assert h.attributes.mass.mass_kg.value == 70.0
    assert h.attributes.mass.mass_kg.source == HUMAN_MASS_PRIOR_SOURCE
    assert obj.attributes.mass.mass_kg.value == 2.0
    assert obj.attributes.mass.mass_kg.source == VISUAL_MASS_RUNTIME_SOURCE
    assert group["phi"] == max(h.policy.risk, obj.policy.risk)
    grid = result.field.pipeline.configuration_space.occupancy_grids[0]
    for member in members:
        cells = grid.cell_track_ids == member.attributes.track_id
        assert cells.any()
        assert np.all(grid.cell_risks[cells] == group["phi"])
    assert result.field.all_obstacle_point_count == 24
    assert result.diagnostics["selected_return_count"] == 24
    assert sum(o.diagnostics["policy_return_count"] for o in result.objects) == 24
    assert result.diagnostics["fixed_lambda"] == 1.25


def test_failed_field_transaction_does_not_advance_carried_identity_or_motion(
    monkeypatch,
):
    import robot.evaluation.multi_object_perception as mapping

    runtime = setup(monkeypatch)
    first = runtime(observed_packet(1, 1.0))
    state = runtime.carried_runtime_state
    original = mapping.build_safe_sage_field_pipeline
    monkeypatch.setattr(
        mapping,
        "build_safe_sage_field_pipeline",
        lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("controlled field failure")
        ),
    )
    with pytest.raises(RuntimeError, match="field failure"):
        runtime(observed_packet(2, 1.2))
    assert runtime.carried_runtime_state is state
    assert state.last_times["human"] == 1.0
    assert runtime.previous is first.field
    monkeypatch.setattr(mapping, "build_safe_sage_field_pipeline", original)
    recovered = runtime(observed_packet(3, 1.2))
    assert recovered.field.sensor_timestamp_seconds == 1.2
    assert runtime.carried_runtime_state.last_times["human"] == 1.2


def test_missing_prop_keeps_only_bounded_identity_without_publishing_old_evidence(
    monkeypatch,
):
    import robot.evaluation.carried_object_runtime as bridge

    runtime = setup(monkeypatch)
    for frame, t in enumerate((1.0, 1.2, 1.4), 1):
        result = runtime(observed_packet(frame, t))
    detail = result.diagnostics["carried_object_groups"]
    object_id = detail["members"]["object"]
    original = bridge.propose_carried_object
    monkeypatch.setattr(bridge, "propose_carried_object", lambda *a, **kw: (None, []))
    missing = runtime(observed_packet(4, 1.5))
    assert not missing.diagnostics["carried_object_groups"]["confirmed_groups"]
    assert not any(
        o.diagnostics.get("carried_member") == "object" for o in missing.objects
    )
    assert runtime.carried_runtime_state.last_times["object"] == 1.4
    monkeypatch.setattr(bridge, "propose_carried_object", original)
    recovered = runtime(observed_packet(5, 1.6))
    assert (
        recovered.diagnostics["carried_object_groups"]["members"]["object"] == object_id
    )
    assert recovered.diagnostics["carried_object_groups"]["confirmed_groups"]
    # Beyond .35 s, even identity must restart from fresh observations.
    # The unchanged temporal-field gap gate first requires an unpublished baseline.
    with pytest.raises(ValueError, match="temporal source gap"):
        runtime(observed_packet(6, 2.0))
    later = runtime(observed_packet(7, 2.1))
    assert later.diagnostics["carried_object_groups"]["members"]["object"] != object_id
    assert not later.diagnostics["carried_object_groups"]["confirmed_groups"]
