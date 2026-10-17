"""Separate authored layout monitoring from observed restaurant CRA evidence."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from robot.config import PROJECT_ROOT


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
    return {
        "target_owned_visual_instance":row["accepted_visual_updates"] >= 3
        and diagnostic["accepted_instance"] and diagnostic["policy_return_count"] > 0,
        "target_sensor_mass":mass["value"] is not None and mass["status"] in {"measured","inferred"},
        "target_visual_contact":factors["contact"]["mode"] == "evidence_supported"
        and factors["contact"]["basis"] in {"visual_silhouette_geometry_proxy","visual_exposed_edge_proxy"},
        "target_current_motion":speed["value"] is not None and speed["status"] in {"measured","inferred"},
        "four_numeric_policy_scores":len(factors) == 4 and all(math.isfinite(item["score"]) and 0 <= item["score"] <= 1 for item in factors.values()),
    }


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
