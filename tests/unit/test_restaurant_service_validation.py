import copy
import hashlib
import json
import math

import numpy as np
import pytest

from robot.evaluation.restaurant_service_validation import authored_raster_clearance, restaurant_asset_checks
from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint


def _summary():
    return {"fixtures":{"FoodBowl":{
        "accepted_visual_updates":3,
        "latest":{
            "attributes":{
                "mass":{"mass_kg":{"value":.4,"status":"inferred"}},
                "motion":{"relative_speed_mps":{"value":.01,"status":"inferred"}},
            },
            "diagnostics":{
                "accepted_instance":True,"policy_return_count":16,
                "factor_scores":{
                    "contact":{"mode":"evidence_supported","basis":"visual_silhouette_geometry_proxy","score":.2},
                    "kinetic":{"mode":"evidence_supported","basis":"energy","score":.18},
                    "potential":{"mode":"configured_prior","basis":"unknown","score":.25},
                    "thermal":{"mode":"configured_prior","basis":"unknown","score":.25},
                },
            },
        },
    }}}


def test_real_mesh_gate_accepts_bounded_component_with_explicit_missing_priors():
    assert all(restaurant_asset_checks(_summary()).values())


@pytest.mark.parametrize("problem", ["no_mass","retained_mass","no_instance","no_points","only_two_frames","prior_contact","missing_factor","nonfinite"])
def test_numeric_policies_do_not_replace_real_mesh_component_evidence(problem):
    summary=copy.deepcopy(_summary())
    row=summary["fixtures"]["FoodBowl"]
    item=row["latest"]
    diagnostic=item["diagnostics"]
    if problem=="no_mass":
        item["attributes"]["mass"]["mass_kg"]["value"]=None
    elif problem=="retained_mass":
        item["attributes"]["mass"]["mass_kg"]["status"]="prior"
    elif problem=="no_instance":
        diagnostic["accepted_instance"]=False
    elif problem=="no_points":
        diagnostic["policy_return_count"]=0
    elif problem=="only_two_frames":
        row["accepted_visual_updates"]=2
    elif problem=="prior_contact":
        diagnostic["factor_scores"]["contact"]["mode"]="configured_prior"
    elif problem=="missing_factor":
        del diagnostic["factor_scores"]["thermal"]
    else:
        diagnostic["factor_scores"]["thermal"]["score"]=math.nan
    assert not all(restaurant_asset_checks(summary).values())


def test_missing_target_cannot_pass():
    assert not all(restaurant_asset_checks({"fixtures":{}}).values())


def test_authored_clearance_respects_asymmetric_rotated_footprint_and_pixels():
    footprint=RectangularRobotFootprint(.2,.6,.25,.25,.05)
    points=np.array([[1.0,0.0],[100.,100.]])
    assert authored_raster_clearance(points,(0,0,0),footprint,.04)==pytest.approx(.71)
    assert authored_raster_clearance(points,(0,0,math.pi/2),footprint,.04)==pytest.approx(.66)
    assert authored_raster_clearance(np.array([[0.,0.]]),(0,0,0),footprint,.04)<0
    assert authored_raster_clearance(points,(50,50,0),footprint,.04)==2
    with pytest.raises(ValueError):
        authored_raster_clearance(points,(0,0,math.nan),footprint,.04)


def test_reproducible_survey_clips_floor_and_ceiling_and_checks_fingerprints(tmp_path, monkeypatch):
    from pxr import Usd, UsdGeom
    from PIL import Image
    from scripts.evaluation import survey_restaurant_layout as survey
    from robot.evaluation import restaurant_service_validation as monitor
    monkeypatch.setattr(survey,"ROOT",tmp_path)
    monkeypatch.setattr(monitor,"PROJECT_ROOT",tmp_path)
    scene=tmp_path/"source.usda"
    stage=Usd.Stage.CreateNew(str(scene))
    UsdGeom.SetStageMetersPerUnit(stage,1)
    for name,points in {
        "Wall":[(0,0,0),(1,0,0),(1,0,2),(0,0,2)],
        "Floor":[(2,0,0),(3,0,0),(3,1,0),(2,1,0)],
        "Ceiling":[(4,0,2),(5,0,2),(5,1,2),(4,1,2)],
    }.items():
        mesh=UsdGeom.Mesh.Define(stage,f"/World/{name}")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
    stage.GetRootLayer().Save()
    out=tmp_path/"survey"
    report=survey.export_survey(scene,scene,out)
    assert report["projected_faces"]==1
    assert report["source_files_unchanged"]
    mask=out/"authored-height-column-mask.png"
    pixels=np.array(Image.open(mask))
    assert pixels[int(32/.025),int((.5+2)/.025)]>0
    assert pixels[int((32-.5)/.025),int((2.5+2)/.025)]==0
    assert hashlib.sha256(mask.read_bytes()).hexdigest()==report["mask_sha256"]
    points,allowance,_=monitor.load_authored_layout_monitor(out/"geometry-survey.json",[])
    assert len(points)>0 and allowance>0
    Image.new("L",(1760,1360),0).save(mask)
    with pytest.raises(ValueError,match="fingerprint"):
        monitor.load_authored_layout_monitor(out/"geometry-survey.json",[])
    metadata=out/"geometry-survey.json"
    corrupt=json.loads(metadata.read_text())
    corrupt.pop("mask_sha256")
    metadata.write_text(json.dumps(corrupt))
    scene.write_text(scene.read_text()+"\n# changed source\n")
    with pytest.raises(ValueError,match="source changed"):
        monitor.load_authored_layout_monitor(metadata,[])


def test_named_asset_gate_never_substitutes_a_different_observed_object():
    summary=_summary()
    assert not all(restaurant_asset_checks(summary,"SecondPot").values())
    summary["fixtures"]["SecondPot"]=summary["fixtures"].pop("FoodBowl")
    assert all(restaurant_asset_checks(summary,"SecondPot").values())


def test_restaurant_views_are_finite_robot_setup_only_and_reference_real_props():
    from robot.scene.restaurant_service_build import RESTAURANT_OBSERVATIONS, service_props
    names={prop.name for prop in service_props()}
    for target,position,yaw in RESTAURANT_OBSERVATIONS.values():
        assert target in names
        assert len(position)==3 and all(math.isfinite(v) for v in (*position,yaw))
        assert position[2] == pytest.approx(.07382)


def _plume_record(phase, value, status="inferred", accepted=True, epoch=1):
    from test_object_map_validation import object_row
    row=object_row()
    row["attributes"]={"timestamp_seconds":epoch*.1,
        "thermal":{"heat_source_likelihood":{"value":value,"status":status,"timestamp_seconds":epoch*.1}}}
    row["diagnostics"].update(accepted_instance=accepted, policy_return_count=12,
        factor_scores={"thermal":{"mode":"evidence_supported","basis":"visual_object_plume_proxy"}})
    return {"objects":[row],"restaurant_plume_phase":phase,"sensor_frame_id":epoch,"sensor_timestamp":epoch*.1}


def test_restaurant_plume_gate_requires_supported_continuous_ordering():
    from robot.evaluation.restaurant_service_validation import restaurant_plume_checks
    fixture=[{"name":"FoodBowl","center_world_xy":[1,0]}]
    records=[_plume_record("no_plume",v,epoch=i) for i,v in enumerate([.01,.02,.03],1)]
    records += [_plume_record("visible_plume",v,epoch=i) for i,v in enumerate([.45,.55,.65],4)]
    result=restaurant_plume_checks(records,fixture)
    assert result["passed"]
    assert result["median_cue_scores"]==pytest.approx({"no_plume":.02,"visible_plume":.55})
    for record in records:
        record["objects"][0]["attributes"]["thermal"]["heat_source_likelihood"]["status"]="prior"
    assert not restaurant_plume_checks(records,fixture)["passed"]


@pytest.mark.parametrize("problem",["unavailable","unowned","ambiguous","wrong_basis","no_ordering","too_few","stale","duplicate_epoch","nonfinite"])
def test_restaurant_plume_gate_never_promotes_missing_or_wrong_proxy_evidence(problem):
    from robot.evaluation.restaurant_service_validation import restaurant_plume_checks
    records=[_plume_record(phase,value,epoch=i) for i,(phase,value) in enumerate([("no_plume",.01)]*3+[("visible_plume",.6)]*3,1)]
    for record in records:
        row=record["objects"][0]
        if problem=="unavailable":
            row["attributes"]["thermal"]["heat_source_likelihood"]["value"]=None
        elif problem=="unowned":
            row["diagnostics"]["accepted_instance"]=False
        elif problem=="ambiguous":
            record["objects"].append(copy.deepcopy(row))
        elif problem=="wrong_basis":
            row["diagnostics"]["factor_scores"]["thermal"]["basis"]="physical"
        elif problem=="no_ordering":
            row["attributes"]["thermal"]["heat_source_likelihood"]["value"]=.01
        elif problem=="stale":
            row["attributes"]["thermal"]["heat_source_likelihood"]["timestamp_seconds"]=-1
        elif problem=="duplicate_epoch":
            record["sensor_frame_id"]=1
        elif problem=="nonfinite":
            row["attributes"]["thermal"]["heat_source_likelihood"]["value"]=math.nan
    if problem=="too_few":
        records.pop()
    assert not restaurant_plume_checks(records,[{"name":"FoodBowl","center_world_xy":[1,0]}])["passed"]


@pytest.mark.parametrize("changed",[(.02,0,0),(0,0,.02),(0,0,math.nan)])
def test_fixed_source_replay_rejects_motion_or_invalid_camera_pose(changed):
    from scripts.evaluation.restaurant_stationary_plume_replay import validate_stationary
    with pytest.raises(ValueError):
        validate_stationary([{"stationary_rgb_pose":[0,0,0]},{"stationary_rgb_pose":changed}])
