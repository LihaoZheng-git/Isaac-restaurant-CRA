"""Separate authored layout monitoring from observed restaurant CRA evidence."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from robot.config import PROJECT_ROOT


def restaurant_plume_checks(records, fixtures):
    """Evaluate continuous cue ordering from fresh matched sensor assessments.

    Authored fixture matching is evaluation only. Numeric priors and unknown
    heat never establish a low/no-plume observation.
    """
    from robot.evaluation.object_map_validation import summarize_fixture_assessments
    scores = {"no_plume": [], "visible_plume": []}
    rejected = {name: 0 for name in scores}
    previous_epoch = (-1, -1.)
    for record in records:
        phase = record.get("restaurant_plume_phase")
        if phase not in scores:
            continue
        epoch = (record["sensor_frame_id"], record["sensor_timestamp"])
        if epoch[0] <= previous_epoch[0] or epoch[1] <= previous_epoch[1]:
            rejected[phase] += 1
            continue
        previous_epoch = epoch
        summary = summarize_fixture_assessments([record], fixtures, maximum_distance_m=.20)
        row = summary["fixtures"].get("FoodBowl", {}).get("latest")
        if row is not None:
            heat = row["attributes"]["thermal"]["heat_source_likelihood"]
            diagnostic = row["diagnostics"]
            age = row["attributes"]["timestamp_seconds"] - heat["timestamp_seconds"]
            if (diagnostic["accepted_instance"] and diagnostic["policy_return_count"] > 0
                and heat["value"] is not None and math.isfinite(heat["value"]) and 0 <= heat["value"] <= 1
                and heat["status"] in {"measured", "inferred"}
                and 0 <= age <= .15 + 1e-9
                and diagnostic["factor_scores"]["thermal"]["mode"] == "evidence_supported"
                and diagnostic["factor_scores"]["thermal"]["basis"] == "visual_object_plume_proxy"):
                scores[phase].append(heat["value"])
                continue
        rejected[phase] += 1
    medians = {key: float(np.median(values)) if values else None for key, values in scores.items()}
    low, high = medians["no_plume"], medians["visible_plume"]
    checks = {"fresh_low_cue": len(scores["no_plume"]) >= 3 and low <= .1,
              "fresh_positive_cue": len(scores["visible_plume"]) >= 3 and high > .2,
              "continuous_cue_ordering": low is not None and high is not None and high-low > .1}
    return {"passed": all(checks.values()), "checks": checks,
            "fresh_cue_scores": scores, "median_cue_scores": medians,
            "rejected_or_unavailable_updates": rejected,
            "scope": "stationary RGB visible-plume proxy on real bowl; no temperature, soup classification or moving-view acceptance"}


def restaurant_asset_checks(summary, target_name="FoodBowl"):
    """Bounded real-mesh component gate; numeric priors are not fresh inputs."""
    row = summary["fixtures"].get(target_name, {})
    latest = row.get("latest")
    if latest is None:
        return {"target_uniquely_observed":False}
    mass = latest["attributes"]["mass"]["mass_kg"]
    speed = latest["attributes"]["motion"]["relative_speed_mps"]
    diagnostic = latest["diagnostics"]
    factors = diagnostic["factor_scores"]
    checks = {
        "target_owned_visual_instance":row["accepted_visual_updates"] >= 3
        and diagnostic["accepted_instance"] and diagnostic["policy_return_count"] > 0,
        "target_sensor_mass":mass["value"] is not None and mass["status"] in {"measured","inferred"},
        "target_visual_contact":factors["contact"]["mode"] == "evidence_supported"
        and factors["contact"]["basis"] in {"visual_silhouette_geometry_proxy","visual_exposed_edge_proxy"},
        "target_current_motion":speed["value"] is not None and speed["status"] in {"measured","inferred"},
        "four_numeric_policy_scores":len(factors) == 4 and all(math.isfinite(item["score"]) and 0 <= item["score"] <= 1 for item in factors.values()),
    }
    return {key: bool(value) for key, value in checks.items()}


def restaurant_route_sensor_checks(records):
    """New route-start sensor gate, not a remote tabletop-asset acceptance.

    At least one currently visible geometric track must have three fresh owned
    RGB/LiDAR mass/contact/motion assessments. Other tracks retain numeric priors.
    This permits route coverage development; each featured hazard needs its own
    asset gate before a perception claim or dynamic encounter is introduced.
    """
    counts, latest = {}, {}
    for record in records:
        for item in record.get("objects", []):
            key = item["track_id"]
            row = {"latest": item, "accepted_visual_updates": 3}
            checks = restaurant_asset_checks({"fixtures": {"track": row}}, "track")
            if all(checks.values()):
                counts[key] = counts.get(key, 0)+1
                latest[key] = record["sensor_timestamp"]
    now = records[-1]["sensor_timestamp"] if records else math.inf
    accepted = [key for key in counts if counts[key] >= 3 and 0 <= now-latest[key] <= .35]
    return {"fresh_owned_mass_contact_motion_track": bool(accepted),
            "at_least_twelve_real_frames": len(records) >= 12}


def load_authored_layout_monitor(survey_path, additions):
    """Load a fingerprinted offline survey and conservatively include new props.

    Points are independent evaluation/abort information. They must never enter
    perception, CRA, the nominal, MPC, or CBF. No fresh sensing is claimed.
    """
    path = Path(survey_path)
    survey = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in survey["source_sha256"].items():
        if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("layout monitor source changed; regenerate authored survey")
    mask_path = path.parent / "authored-height-column-mask.png"
    if "mask_sha256" in survey and hashlib.sha256(mask_path.read_bytes()).hexdigest() != survey["mask_sha256"]:
        raise ValueError("authored layout mask fingerprint mismatch")
    image = Image.open(mask_path).convert("L")
    draw = ImageDraw.Draw(image)
    xmin,xmax,ymin,ymax = survey["raster_limits_xmin_xmax_ymin_ymax"]
    resolution = survey["raster_resolution_m"]
    for item in additions:
        low,high = item["bounds_minimum"],item["bounds_maximum"]
        draw.rectangle(((low[0]-xmin)/resolution,(ymax-high[1])/resolution,
                        (high[0]-xmin)/resolution,(ymax-low[1])/resolution),fill=255)
    rows,cols = np.nonzero(np.asarray(image)>0)
    points = np.column_stack((xmin+(cols+.5)*resolution,ymax-(rows+.5)*resolution))
    return points, math.sqrt(2)*resolution, (xmin,xmax,ymin,ymax)


def authored_raster_clearance(points, state_world, footprint, pixel_allowance):
    """Conservative occupied-pixel to asymmetric rectangle distance (oracle)."""
    state = np.asarray(state_world,dtype=float)
    if state.shape != (3,) or not np.isfinite(state).all():
        raise ValueError("monitor state must be finite x,y,yaw")
    delta = np.asarray(points)-state[:2]
    # Bounding radius proves excluded points are at least 2 m from the base.
    radius = math.hypot(max(footprint.front_extent,footprint.rear_extent),
                        max(footprint.left_extent,footprint.right_extent))+footprint.safety_margin
    nearby = delta[np.linalg.norm(delta,axis=1) <= radius+2+pixel_allowance]
    if not len(nearby):
        return 2.0
    c,s = math.cos(state[2]),math.sin(state[2])
    local = nearby @ np.array([[c,-s],[s,c]])
    margin = footprint.safety_margin
    nearest = np.clip(local,[-footprint.rear_extent-margin,-footprint.right_extent-margin],
                      [footprint.front_extent+margin,footprint.left_extent+margin])
    return float(np.linalg.norm(local-nearest,axis=1).min()-pixel_allowance)
