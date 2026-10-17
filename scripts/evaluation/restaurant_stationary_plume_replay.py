"""Fixed-camera RGB cue replay with an explicit stationary source-region PRIOR.

This never repairs live track IDs, refreshes measured geometry or commands
wheels. The original registered-method rejection remains a separate result.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robot.evaluation.object_map_validation import summarize_fixture_assessments
from robot.perception.camera_lidar_fusion import TrackSemanticAssociation
from robot.perception.object_plume_evidence import ObjectPlumeObservation, estimate_object_plume_evidence
from robot.perception.physical_attributes import EstimateStatus, PhysicalAttributes, ScalarEstimate
from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection, SemanticDetectionFrame
from robot.perception.specialist_track_evidence import SpecialistAssociationFrame
from robot.risk.object_risk_policy import ObjectRiskPriorConfig, evaluate_object_risk
from robot.risk.physical_factor_adapter import PhysicalFactorAdapterConfig, adapt_physical_factors


def validate_stationary(records):
    poses = np.array([row["stationary_rgb_pose"] for row in records], dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3 or not np.isfinite(poses).all():
        raise ValueError("missing finite copied stationary poses")
    yaw = (poses[:, 2]-poses[0, 2]+np.pi) % (2*np.pi)-np.pi
    if np.linalg.norm(poses[:, :2]-poses[0, :2], axis=1).max() > .01 or abs(yaw).max() > .01:
        raise ValueError("fixed-source component requires a stationary camera")


def plot_replay(source, report, records, box):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, phase in zip(axes[:2], ("no_plume", "visible_plume")):
        index = max(i for i, row in enumerate(records) if row["restaurant_plume_phase"] == phase)
        rgb = np.array(Image.open(source/"plume-rgb"/f"{index+1:03d}.png").convert("RGB"))
        axis.imshow(rgb)
        axis.add_patch(Rectangle((box.x_min,box.y_min),box.width,box.height,fill=False,color="#39a95b",linewidth=1))
        axis.set_xlim(max(0,box.x_min-35),min(rgb.shape[1],box.x_max+35))
        axis.set_ylim(min(rgb.shape[0],box.y_max+20),max(0,box.y_min-70))
        axis.set_title("Recorded RGB: " + phase.replace("_"," "))
        axis.axis("off")
    for phase, color in (("no_plume","#4478aa"),("visible_plume","#d78035")):
        axes[2].plot(report["phases"][phase]["cue_scores"],label=phase.replace("_"," "),color=color)
    axes[2].set(ylim=(-.02,1),ylabel="Continuous visible-plume cue",xlabel="Usable window index in each phase")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=.2)
    fig.suptitle("Stationary restaurant RGB component: explicit source-region PRIOR")
    fig.supxlabel("Green source box derives from an earlier accepted SAM instance and is retained as a PRIOR.\nRegistered live association failed; this is offline fixed-camera evaluation, not temperature or navigation acceptance.",fontsize=9)
    fig.savefig(source/"stationary-plume-component.png",dpi=150)
    plt.close(fig)


def replay(source):
    source = Path(source)
    records = json.loads((source/"stationary-fields.json").read_text())
    gate = json.loads((source/"stationary-gate.json").read_text())
    if gate["nonzero_wheel_commands"] != 0:
        raise ValueError("stationary component cannot consume wheel-motion records")
    validate_stationary(records)
    fixtures = json.loads((source/"fixture-monitor.json").read_text())["fixtures"]
    candidate = None
    for index, record in enumerate(records):
        if record["restaurant_plume_phase"] != "clean_reference_validation":
            continue
        row = summarize_fixture_assessments([record], fixtures, .20)["fixtures"]["FoodBowl"]["latest"]
        if row and row["diagnostics"]["accepted_instance"]:
            candidate = (index, record, row)
    if candidate is None:
        raise ValueError("no accepted observed source box in the clean-reference phase")
    index, reference_record, source_row = candidate
    diagnostic = source_row["diagnostics"]
    if diagnostic["instance_mask_pixels"] < 200:
        raise ValueError("observed source mask is unresolved")
    box = PixelBoundingBox(*diagnostic["instance_box_xyxy"])
    reference = np.array(Image.open(source/"plume-rgb"/f"{index+1:03d}.png").convert("RGB"))
    height, width = reference.shape[:2]
    mask = np.zeros((height, width), bool)
    mask[int(box.y_min):int(np.ceil(box.y_max)), int(box.x_min):int(np.ceil(box.x_max))] = True
    reference_time = source_row["attributes"]["timestamp_seconds"]
    output = {}
    for phase in ("no_plume", "visible_plume"):
        observations, values, thermal_scores, phis, rejections = [], [], [], [], []
        for frame_index, record in enumerate(records):
            if record["restaurant_plume_phase"] != phase:
                continue
            timestamp = record["objects"][0]["attributes"]["timestamp_seconds"]
            frame_id = record["camera_frame_id"]
            rgb = np.array(Image.open(source/"plume-rgb"/f"{frame_index+1:03d}.png").convert("RGB"))
            # This is a newly evaluated configuration PRIOR for the declared
            # unchanged source region, not a refreshed sensor detection/track.
            detection = SemanticDetection(0, "stationary_source_region_prior", .5, box, mask)
            frame = SemanticDetectionFrame(frame_id, timestamp, height, width, (detection,),
                "stationary source-region PRIOR; original observed box timestamp retained in report", 0.)
            link = TrackSemanticAssociation(source_row["track_id"], "matched", 0, 0, 0, 1., 1., True, box.center_xy)
            observations.append(ObjectPlumeObservation(rgb, frame,
                SpecialistAssociationFrame(frame_id, timestamp, height, width, (link,)),
                timestamp, True, EstimateStatus.PRIOR))
            observations = [item for item in observations if timestamp-item.detections.source_timestamp_seconds <= .5]
            result = estimate_object_plume_evidence(reference, reference_timestamp_seconds=reference_time,
                observations=tuple(observations), timestamp_seconds=timestamp, clean_reference_validated=True,
                negative_validation_scope="explicitly clean, fixed-camera static restaurant bowl region PRIOR; visible cue only")
            heat = result[source_row["track_id"]].thermal
            if not heat.heat_source_likelihood.is_available:
                rejections.append(list(heat.heat_source_likelihood.evidence))
                continue
            if heat.heat_source_likelihood.status is not EstimateStatus.PRIOR:
                raise RuntimeError("retained source-region prior provenance was lost")
            heat = replace(heat, heat_source_likelihood=replace(heat.heat_source_likelihood,
                evidence=(*heat.heat_source_likelihood.evidence,
                    f"stationary source-region PRIOR from original observation t={reference_time}; original track={source_row['track_id']}; no measured source timestamp renewed")))
            # Factor-specific offline evaluation: other factors retain .25
            # priors, and unit interaction is an explicit matched PRIOR.
            interaction = ScalarEstimate(1., 1., 1., .5, "1", "offline_unit_interaction_prior",
                timestamp, EstimateStatus.PRIOR, ("factor-specific diagnostic only; no current route/navigation claim",))
            attributes = replace(PhysicalAttributes.unknown(track_id=source_row["track_id"],timestamp_seconds=timestamp),thermal=heat)
            adapted = adapt_physical_factors(attributes,timestamp_seconds=timestamp,
                thermal_interaction_likelihood=interaction,thermal_basis="visual_object_plume_proxy",
                config=PhysicalFactorAdapterConfig(allow_visual_thermal_proxy=True))
            policy, detail = evaluate_object_risk(adapted,ObjectRiskPriorConfig())
            values.append(heat.heat_source_likelihood.value)
            thermal_scores.append(detail["thermal"]["score"])
            phis.append(policy.risk)
            last_heat = asdict(heat.heat_source_likelihood)
        output[phase] = {"cue_scores":values,"thermal_factor_scores":thermal_scores,"phi":phis,
            "last_heat":last_heat if values else None,"rejections":rejections}
    low, high = output["no_plume"]["cue_scores"], output["visible_plume"]["cue_scores"]
    passed = len(low)>=3 and len(high)>=3 and np.median(low)<=.1 and np.median(high)>np.median(low)+.1
    report = {"passed":bool(passed),"phases":output,
        "source_original_timestamp":reference_time,"source_original_track_id":source_row["track_id"],
        "source_box_xyxy":diagnostic["instance_box_xyxy"],"source_region_status":"prior",
        "scope":"stationary RGB component with declared source-region/identity PRIOR, offline unit interaction PRIOR and .25 other-factor priors; no live identity, temperature or wheel-motion acceptance"}
    (source/"fixed-source-plume-replay.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    plot_replay(source, report, records, box)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--static-source-region-prior",action="store_true")
    parser.add_argument("--clean-reference-validated",action="store_true")
    args=parser.parse_args()
    if not args.static_source_region_prior or not args.clean_reference_validated:
        parser.error("explicit fixed-source prior and clean-reference validation are required")
    report=replay(args.source)
    print(json.dumps({"passed":report["passed"],"source_region_status":"prior",
        "median_cues":{k:float(np.median(v["cue_scores"])) if v["cue_scores"] else None for k,v in report["phases"].items()}}))
    if not report["passed"]:
        raise RuntimeError("fixed-source stationary component did not establish cue ordering")


if __name__=="__main__":
    main()
