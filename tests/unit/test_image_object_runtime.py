from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from test_live_object_perception import packet
from test_object_risk_map import mapped_worker, scalar
from test_image_object_association import proposal
from robot.perception.image_object_association import ImageObjectConfig
from robot.perception.physical_attributes import MassAttributes


def setup():
    runtime = mapped_worker()
    runtime.image_object_config = ImageObjectConfig()
    runtime.grid_config = replace(runtime.grid_config, resolution=.1)
    def proposals(ids, projection):
        mask = np.zeros((100, 100), bool)
        xy = projection.pixel_indices_xy
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        mask[max(1,lo[1]-2):min(99,hi[1]+3), max(1,lo[0]-2):min(99,hi[0]+3)] = True
        return (proposal(mask),)
    runtime.segmenter = SimpleNamespace(set_image=lambda rgb: None, image_proposals=proposals)
    runtime.mass_estimator = SimpleNamespace(estimate_instance=lambda *a, **kw:
        MassAttributes(scalar(2, "kg", kw["timestamp_seconds"])))
    return runtime


def observed(frame, time):
    xyz = np.array([[x,y,z] for x in (2., 2.03) for y in (-.25,-.22,.22,.25) for z in (.2,.23)])
    transform = np.eye(4)
    transform[:3,:3] = [[0,-1,0],[0,0,1],[-1,0,0]]
    return replace(packet(frame,time), obstacle_xyz=xyz, scene_xyz=xyz, camera_from_scan_robot=transform)


def test_runtime_applies_group_policy_only_to_current_points_without_raw_track_merge():
    runtime = setup()
    for i,t in enumerate((1.,1.1,1.2),1):
        result = runtime(observed(i,t))
    groups = [o for o in result.objects if o.diagnostics.get("image_object_association")]
    assert len(groups) == 1
    assert groups[0].attributes.mass.mass_kg.value == 2
    assert result.field.all_obstacle_point_count == 16
    assert len(result.field.tracking_result.tracked_obstacles) == 2
    assert max(result.field.tracking_result.point_track_ids) < 5_000_000_000
    assert sum(o.diagnostics["policy_return_count"] for o in result.objects) == 16
    assert result.diagnostics["fixed_lambda"] == 1.25


def test_local_contact_flows_into_group_cra_without_geometry_or_prior_changes():
    from robot.perception.local_contact_geometry import LocalContactGeometryConfig
    from robot.perception.visual_shape_contact import SilhouetteContactConfig
    groups, fields = [], []
    for local in (False, True):
        runtime = setup()
        # This contact integration fixture must exceed the unchanged 200-pixel
        # silhouette gate; the smaller association-only fixture has 174 pixels.
        def resolved_proposal(ids, projection):
            mask = np.zeros((100, 100), bool)
            mask[35:45, 34:67] = True
            return (proposal(mask),)
        runtime.segmenter.image_proposals = resolved_proposal
        if local:
            runtime.shape_contact_config = SilhouetteContactConfig(local_geometry=LocalContactGeometryConfig())
        for i, t in enumerate((1., 1.1, 1.2), 1):
            result = runtime(observed(i, t))
        group = next(o for o in result.objects if o.diagnostics.get("image_object_association"))
        groups.append(group)
        fields.append(result.field)
        proxy = group.diagnostics["contact_proxy"]
        assert proxy["timestamp_seconds"] == pytest.approx(1.3)
        assert any(("local_convex_multiscale" if local else "legacy_global_outline") in e for e in proxy["evidence"])
        assert result.diagnostics["fixed_lambda"] == 1.25
        assert group.diagnostics["factor_scores"]["contact"]["mode"] == "evidence_supported"
    assert groups[0].diagnostics["contact_proxy"]["value"] != groups[1].diagnostics["contact_proxy"]["value"]
    assert groups[0].diagnostics["factor_scores"]["contact"]["score"] != groups[1].diagnostics["factor_scores"]["contact"]["score"]
    for factor in ("kinetic", "potential", "thermal"):
        assert groups[0].diagnostics["factor_scores"][factor] == groups[1].diagnostics["factor_scores"][factor]
    assert groups[0].attributes.mass == groups[1].attributes.mass
    assert fields[0].all_obstacle_point_count == fields[1].all_obstacle_point_count == 16
    np.testing.assert_array_equal(fields[0].tracking_result.point_track_ids, fields[1].tracking_result.point_track_ids)


def test_field_failure_does_not_commit_group_history(monkeypatch):
    import robot.evaluation.multi_object_perception as mapping
    runtime = setup()
    first = runtime(observed(1,1.))
    state = runtime.image_object_state
    original = mapping.build_safe_sage_field_pipeline
    def fail(*a,**kw):
        raise RuntimeError("controlled field failure")
    monkeypatch.setattr(mapping, "build_safe_sage_field_pipeline", fail)
    with pytest.raises(RuntimeError, match="field failure"):
        runtime(observed(2,1.1))
    assert runtime.image_object_state is state and state.last_time == 1.
    assert runtime.previous is first.field
    monkeypatch.setattr(mapping, "build_safe_sage_field_pipeline", original)
    runtime(observed(3,1.1))
    assert runtime.image_object_state.last_time == 1.1


def test_refinement_changes_only_contact_after_group_qualification():
    import cv2
    from robot.perception.contact_mask_refinement import ContactMaskRefinementConfig
    from robot.perception.local_contact_geometry import LocalContactGeometryConfig
    from robot.perception.visual_shape_contact import SilhouetteContactConfig
    class Backend:
        calls = 0
        def set_image(self,rgb):
            self.shape = rgb.shape[:2]
        def predict(self,point,*,context_box_xyxy):
            self.calls += 1
            x0,y0,x1,y1 = context_box_xyxy
            mask = np.zeros(self.shape,np.uint8)
            cy = (y0+y1-1)//2
            cv2.rectangle(mask,(x0+4,y0),(x1-5,y1-1),1,-1)
            for cx in (x0+4,x1-5):
                cv2.circle(mask,(cx,cy),4,1,-1)
            return mask[None].astype(bool),np.array([.95])
    groups,fields = [],[]
    for refinement in (False,True):
        runtime = setup()
        runtime.shape_contact_config = SilhouetteContactConfig(local_geometry=LocalContactGeometryConfig())
        runtime.segmenter.backend = Backend()
        if refinement:
            runtime.contact_mask_refinement_config = ContactMaskRefinementConfig()
        def resolved_proposal(ids,projection):
            mask = np.zeros((100,100),bool)
            mask[35:45,34:67] = True
            return (proposal(mask),)
        runtime.segmenter.image_proposals = resolved_proposal
        for i,t in enumerate((1.,1.1,1.2),1):
            result = runtime(observed(i,t))
            if i < 3:
                assert runtime.segmenter.backend.calls == 0
        groups.append(next(o for o in result.objects if o.diagnostics.get("image_object_association")))
        fields.append(result.field)
        assert runtime.segmenter.backend.calls == int(refinement)
        assert result.diagnostics["fixed_lambda"] == 1.25
    audit = groups[1].diagnostics["contact_mask_refinement"]
    assert audit["accepted"] and audit["source_timestamp_seconds"] == 1.3
    assert groups[0].diagnostics["contact_proxy"]["value"] != groups[1].diagnostics["contact_proxy"]["value"]
    assert groups[0].diagnostics["factor_scores"]["contact"]["score"] != groups[1].diagnostics["factor_scores"]["contact"]["score"]
    for factor in ("kinetic","potential","thermal"):
        assert groups[0].diagnostics["factor_scores"][factor] == groups[1].diagnostics["factor_scores"][factor]
    assert groups[0].attributes.mass == groups[1].attributes.mass
    assert groups[0].diagnostics["instance_mask_pixels"] == groups[1].diagnostics["instance_mask_pixels"] == 330
    np.testing.assert_array_equal(fields[0].tracking_result.point_track_ids,fields[1].tracking_result.point_track_ids)


def test_evaluation_coalesces_only_fresh_verified_members_and_retains_competitors():
    from robot.evaluation.image_object_runtime import coalesce_image_members_for_evaluation
    raw = [{"track_id": k, "diagnostics": {"accepted_instance": False}} for k in (1,2,3)]
    group = {"track_id": 5_000_000_000, "diagnostics": {"accepted_instance": True, "policy_return_count": 8,
        "image_object_association": {"observations": 3, "current_source_time": 1., "rejections": [],
                                     "fragments": [{"raw_track_id": 1}, {"raw_track_id": 2}]}}}
    record = {"sensor_timestamp": 1., "objects": [*raw, group]}
    result = coalesce_image_members_for_evaluation([record])
    assert [o["track_id"] for o in result[0]["objects"]] == [3, 5_000_000_000]
    assert len(record["objects"]) == 4
    assert len(coalesce_image_members_for_evaluation([{**record, "sensor_timestamp": 1.1}])[0]["objects"]) == 4
