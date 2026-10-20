"""Explicit live comparison policies; CRA assessments stay unchanged.

The geometric controller is a circular velocity-aware CBF reference, not an
MPC/Safe-SAGE ablation. Fixed-class versus CRA changes only field boundary data.
Unrecognized SAM instances keep the unknown class; they are never assigned the
low-risk generic object class merely because a mask exists.
"""

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from robot.evaluation.cra_experiments import CRABaseline, PRIMARY_BASELINES
from robot.perception.physical_attributes import EstimateStatus
from robot.safety.cbf_filter import CBFConfig, CBFSafetyFilter
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidancePolicy


def validate_baseline(value):
    baseline = CRABaseline.CONTINUOUS_CRA_SAFE_SAGE if value is None else CRABaseline(value)
    if baseline not in PRIMARY_BASELINES:
        raise ValueError("live primary comparison requires a primary baseline")
    return baseline


def crossing_due(*, now, robot_y, trigger_y, delay_seconds, navigation_start):
    """Actor schedule only; no actor truth or hazard scores enter control."""
    if delay_seconds is None:
        return robot_y >= trigger_y
    return navigation_start is not None and now - navigation_start >= delay_seconds


@lru_cache(maxsize=1)
def fixed_class_comparison_policy():
    # The legacy runtime guidance defaults have a different 1..6 flux scale.
    # Never inherit that scale into the .5..1.7 CRA comparison.
    path = Path(__file__).resolve().parents[2] / "configs/safe_sage_paper_baseline.json"
    policy = SafeSAGEGuidancePolicy.from_json(path)
    mapping = policy.risk_flux_mapping
    if not (mapping.minimum_magnitude == .5 and mapping.maximum_magnitude == 1.7):
        raise ValueError("comparison flux mapping must remain .5..1.7")
    return policy


def applied_field_policies(baseline, policies, objects, *, envelope_origins=None):
    """Return field guidance and an audit without modifying physical evidence."""
    baseline = validate_baseline(baseline)
    class_policy = fixed_class_comparison_policy() if baseline is CRABaseline.FIXED_CLASS_SAFE_SAGE else None
    classes = {}
    for item in objects:
        # This is the existing fresh, unambiguous RGB/SAM person association.
        # The configured numerical adult mass is not consulted.
        cue = item.diagnostics.get("geometry_evidence", {}).get("human_mass_prior", {})
        if (item.diagnostics.get("accepted_instance") and cue.get("value") is not None
                and cue.get("status") == EstimateStatus.PRIOR):
            classes[item.attributes.track_id] = "person"
    guidance, audit = {}, []
    for key, value in policies.items():
        semantic_class = classes.get((envelope_origins or {}).get(key, key), "unknown")
        parameters = (
            class_policy.parameters_for(semantic_class)
            if baseline is CRABaseline.FIXED_CLASS_SAFE_SAGE
            else value.guidance_parameters
        )
        guidance[key] = parameters
        if (not math.isclose(parameters.boundary_flux, -(.5 + 1.2 * parameters.risk), abs_tol=1e-12)
                or parameters.tangential_strength != 1.25):
            raise ValueError("primary comparison requires the shared .5..1.7 flux scale and lambda 1.25")
        audit.append({
            "track_or_group_id": key,
            "observed_class": semantic_class,
            "class_basis": ("fresh_person_sam_association_via_geometry_prior_parent" if key in (envelope_origins or {}) else "fresh_person_sam_association") if semantic_class == "person" else "unavailable_class",
            **({"geometry_prior_parent_track_id": envelope_origins[key]} if key in (envelope_origins or {}) else {}),
            "cra_phi": value.risk,
            "applied_field_risk": parameters.risk,
            "boundary_flux": parameters.boundary_flux,
            "lambda": parameters.tangential_strength,
            "field_used_for_control": baseline is not CRABaseline.GEOMETRIC_VELOCITY_CBF,
        })
    return guidance, audit


def geometric_cbf_command(field, grid, current, nominal, footprint, now, *,
                          maximum_age=.35, maximum_linear_velocity=.2, alpha=1.):
    """Use the same unexpanded occupied cells and tracker velocities.

    Cell half-diagonal and the robot's circumscribed footprint are conservative
    radius allowances. Supported cells use a bounded constant-velocity
    prediction over the actual source age, which is never renewed.
    Unknown/remembered occupancy has an explicit zero-velocity PRIOR. This CBF
    preserves nominal angular velocity and may stop without finding a detour.
    """
    age = float(now - field.sensor_timestamp_seconds)
    if not math.isfinite(age) or not 0 <= age <= maximum_age:
        return np.zeros(2), {"stop_reason": "stale_field", "cbf_margin": None}
    rows, cols = np.nonzero(grid.occupied_mask)
    if not len(rows):
        return np.zeros(2), {"stop_reason": "empty_occupancy", "cbf_margin": None}
    points = np.column_stack((
        grid.config.minimum_x + (cols + .5) * grid.config.resolution,
        grid.config.minimum_y + (rows + .5) * grid.config.resolution,
    ))
    ids = grid.cell_track_ids[rows, cols]
    world_to_anchor = np.array([
        [math.cos(field.anchor_yaw), -math.sin(field.anchor_yaw)],
        [math.sin(field.anchor_yaw), math.cos(field.anchor_yaw)],
    ])
    velocities = np.zeros_like(points)
    supported = np.zeros(len(points), dtype=bool)
    from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE
    velocity_tracks = [*field.tracking_result.tracked_obstacles,
        *(s.tracked_obstacle for s in getattr(getattr(field, "label_result", None), "semantic_obstacles", ()) if s.label_source == HUMAN_ENVELOPE_SOURCE)]
    for track in velocity_tracks:
        selected = ids == track.track_id
        if track.observation_count >= 2:
            velocities[selected] = np.asarray(track.velocity_world_xy) @ world_to_anchor
            supported[selected] = True
    points += velocities * age
    c, s = math.cos(current[2]), math.sin(current[2])
    anchor_to_robot = np.array([[c, -s], [s, c]])
    points = (points - current[:2]) @ anchor_to_robot
    velocities = velocities @ anchor_to_robot
    radius = math.hypot(
        max(footprint.front_extent, footprint.rear_extent) + footprint.safety_margin,
        max(footprint.left_extent, footprint.right_extent) + footprint.safety_margin,
    ) + grid.config.resolution / math.sqrt(2)
    result = CBFSafetyFilter(CBFConfig(
        safety_radius=radius, alpha=alpha,
        maximum_linear_velocity=maximum_linear_velocity,
    )).filter_command(*nominal, points, velocities)
    margin = float(np.min(2 * np.sum(points * velocities, axis=1)
                          - 2 * points[:, 0] * result.safe_linear_velocity
                          + alpha * (np.sum(points**2, axis=1) - radius**2)))
    safe = result.constraint_feasible and not result.emergency_stop_active and margin >= -1e-9
    return (np.array([result.safe_linear_velocity, result.safe_angular_velocity]) if safe else np.zeros(2)), {
        "stop_reason": None if safe else "geometric_cbf_fail_closed",
        "cbf_margin": margin,
        "safety_radius_m": radius,
        "occupied_cell_count": len(points),
        "current_velocity_cells": int(supported.sum()),
        "zero_velocity_prior_cells": int((~supported).sum()),
        "source_age_seconds": age,
        "constant_velocity_prediction_seconds": age,
        "intervention_active": result.intervention_active,
        "scope": "circular velocity-aware CBF; nominal omega preserved; no Safe-SAGE field/MPC command",
    }
