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
