from copy import deepcopy

import pytest

from scripts.evaluation.cra_category_free_report import (
    summarize_capture,
    summarize_motion,
)
from robot.perception.sam_instance_segmentation import SAM_INSTANCE_SOURCE
from robot.perception.visual_mass_runtime import (
    CATEGORY_FREE_MASS_EVIDENCE,
    VISUAL_MASS_RUNTIME_SOURCE,
)


def inputs():
    factors = {
        name: {
            "available": True,
            "nominal_risk": 0.1,
            "conservative_risk": 0.7,
            "confidence": 0.3,
            "basis": "physical",
            "rejection_reasons": [],
            "inputs": [{"status": "prior" if name == "potential" else "inferred"}],
        }
        for name in ("potential", "kinetic", "contact", "thermal")
    }
    report = {
        key: True
        for key in (
            "stationary_passed",
            "mapping_passed",
            "sensor_dataflow_passed",
            "sam_instance_proposal_enabled",
            "category_free_visual_mass_enabled",
            "instance_mask_geometry_enabled",
            "root_layer_unchanged_by_fixture",
            "motion_gate_passed",
        )
    }
    report.update(
        {
            "profiles": {},
            "sphere_density_prior_kg_m3": None,
            "fixed_lambda": 1.25,
            "sensor_phi": 0.98,
            "unassessed_occupancy_return_count": 10,
            "sensor_attributes": {
                "semantic_source": SAM_INSTANCE_SOURCE,
                "semantic_class": "unknown",
                "mass": {
                    "mass_kg": {
                        "source": VISUAL_MASS_RUNTIME_SOURCE,
                        "status": "inferred",
                        "value": 1.0,
                        "confidence": 0.25,
                        "evidence": [CATEGORY_FREE_MASS_EVIDENCE],
                    }
                },
                "geometry": {"fall_height_m": {"value": 0.3, "status": "inferred"}},
                "motion": {"relative_speed_mps": {"value": 0.2}},
            },
            "sensor_factors": {
                "factor_specific": factors,
                "aggregate": {"nominal_phi": 0.3},
            },
            "runtime_perception_readiness": {
                "ready": True,
                "evidence_mode": "prior",
                "scope": "four_factor",
            },
            "prior_assisted_perception_readiness": {
                "ready": True,
                "evidence_mode": "prior",
                "scope": "four_factor",
            },
            "policies": {
                "observed": {"safe_sage": {"risk": 0.98, "tangential_strength": 1.25}}
            },
        }
    )
    sam = {
        "source": SAM_INSTANCE_SOURCE,
        "rejection_reasons": [],
        "prior_detection_mask_overlaps": [{"class": "chair", "overlap_fraction": 0.0}],
    }
    return report, {"live_dataflow_valid": True}, sam


def test_actual_detector_miss_and_prior_mode_are_separate_from_unknown_evidence():
    row = summarize_capture(*inputs())
    assert row["accepted"]
    assert row["yolo_mask_miss_on_sam_instance"] is True
    assert row["category"] == "unknown" and not row["unavailable_factors"]
    assert row["evidence_mode"] == "prior"


@pytest.mark.parametrize(
    "case",
    [
        "sensor_failure",
        "density",
        "oracle",
        "lambda",
        "changed_phi",
        "unknown_factor",
        "wrong_mode",
        "mass_prior",
        "rejected_mask",
        "bad_mass_flow",
    ],
)
def test_incomplete_or_misreported_runs_cannot_pass(case):
    report, dataflow, sam = deepcopy(inputs())
    if case == "sensor_failure":
        report["stationary_passed"] = False
    elif case == "density":
        report["sphere_density_prior_kg_m3"] = [25, 5, 100]
    elif case == "oracle":
        report["profiles"] = {"low": {"potential": 0}}
    elif case == "lambda":
        report["fixed_lambda"] = 1.3
    elif case == "changed_phi":
        report["policies"]["observed"]["safe_sage"]["risk"] = 0.5
    elif case == "unknown_factor":
        report["sensor_factors"]["factor_specific"]["potential"]["available"] = False
    elif case == "wrong_mode":
        report["runtime_perception_readiness"]["evidence_mode"] = "sensor"
    elif case == "mass_prior":
        report["sensor_attributes"]["mass"]["mass_kg"]["status"] = "prior"
    elif case == "rejected_mask":
        sam["rejection_reasons"] = ["ambiguous mask"]
    elif case == "bad_mass_flow":
        dataflow["live_dataflow_valid"] = False
    assert not summarize_capture(report, dataflow, sam)["accepted"]


def test_missing_baseline_overlap_is_not_claimed_as_a_detector_miss():
    report, dataflow, sam = inputs()
    del sam["prior_detection_mask_overlaps"]
    assert (
        summarize_capture(report, dataflow, sam)["yolo_mask_miss_on_sam_instance"]
        is None
    )


def test_recognized_instance_is_not_a_detector_miss():
    report, dataflow, sam = inputs()
    sam["prior_detection_mask_overlaps"][0]["overlap_fraction"] = 0.9
    assert (
        summarize_capture(report, dataflow, sam)["yolo_mask_miss_on_sam_instance"]
        is False
    )


def motion_report():
    return {
        "passed": True,
        "scene_file_unchanged": True,
        "motion": {
            "observed": {
                "passed": True,
                "route_complete": True,
                "motion_mode": "Isaac Nova Carter physical wheel commands",
                "minimum_clearance_m": 0.2,
                "minimum_mpc_margin": 0.0,
                "minimum_cbf_margin": 0.0,
                "mpc_intervention_count": 5,
                "commands": [[0.1, 0.2]],
                "nominal_commands": [[0.2, 0.0]],
                "trajectory": [[0.0, 0.0, 0.0], [2.4, -0.1, 0.2]],
                "stopping": {"passed": True, "minimum_clearance_m": 0.3},
            }
        },
    }


def test_actual_wheel_motion_arrival_and_stopping_are_required():
    assert summarize_motion(motion_report())["accepted"]


@pytest.mark.parametrize(
    "case",
    [
        "not_complete",
        "predicted_only",
        "stopping",
        "nan_stopping",
        "clearance",
        "cbf",
        "no_intervention",
        "invalid_state",
        "zero_commands",
        "scene_changed",
    ],
)
def test_motion_failure_cannot_be_hidden_by_positive_clearance(case):
    report = motion_report()
    trial = report["motion"]["observed"]
    if case == "not_complete":
        trial["route_complete"] = False
    elif case == "predicted_only":
        trial["motion_mode"] = "offline predicted chassis states"
    elif case == "stopping":
        trial["stopping"]["passed"] = False
    elif case == "nan_stopping":
        trial["stopping"]["minimum_clearance_m"] = float("nan")
    elif case == "clearance":
        trial["minimum_clearance_m"] = 0.01
    elif case == "cbf":
        trial["minimum_cbf_margin"] = -0.01
    elif case == "no_intervention":
        trial["mpc_intervention_count"] = 0
    elif case == "invalid_state":
        trial["trajectory"][0][0] = float("nan")
    elif case == "zero_commands":
        trial["commands"] = [[0.0, 0.0]]
    elif case == "scene_changed":
        report["scene_file_unchanged"] = False
    assert not summarize_motion(report)["accepted"]
