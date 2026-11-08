"""Offline boundary-response probes on archived placements, never runtime input."""

from dataclasses import replace
import math

import numpy as np

from robot.safety.class_safety_policy import ClassSafetyParameters, ClassSafetyPolicy
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig, build_local_occupancy_grid
from robot.safety.poisson_safety_function import PoissonSafetyFunctionConfig
from robot.safety.safe_sage_configuration_space import SafeSAGEConfigurationSpaceConfig
from robot.safety.safe_sage_field_pipeline import SafeSAGEFieldPipelineConfig, build_safe_sage_field_pipeline
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidanceParameters, SafeSAGEGuidancePolicy
from robot.safety.safe_sage_h_full import SafeSAGEHFullConfig
from robot.safety.tangential_laplace_guidance_field import TangentialLaplaceGuidanceFieldConfig


def mapped_policies(record, unit_flux):
    """Change only positively identified calibrated object policies.

    Geometry-prior policies deliberately retain their archived parameters. The
    returned map is a memoryless field probe, not a replay of how changing the
    mapping would have changed historical memory admission.
    """
    if isinstance(unit_flux, bool) or not math.isfinite(unit_flux) or unit_flux < 1.7:
        raise ValueError("unit flux must be finite and at least the original1.7")
    changed = {}
    for obj in record["objects"]:
        assessment = obj["diagnostics"].get("restaurant_calibration")
        if assessment and (assessment["eligible"] or assessment["verified_human"]):
            changed[obj["track_id"]] = obj["cra_control"]["safe_sage"]
    if (record["diagnostics"].get("carried_object_groups") or {}).get("confirmed_groups"):
        raise ValueError("carried group replay is outside this fixture scope")
    policies, audit = {}, []
    for row in record["diagnostics"]["applied_field_policies"]:
        key, phi, old = row["track_or_group_id"], row["applied_field_risk"], row["boundary_flux"]
        if key in policies:
            raise ValueError("duplicate applied policy")
        new = old
        if key in changed:
            individual = changed[key]
            if not (math.isclose(phi, individual["risk"], abs_tol=1e-12)
                    and math.isclose(old, individual["boundary_flux"], abs_tol=1e-12)):
                raise ValueError("calibrated individual differs from applied policy")
            if not math.isclose(old, -(.5 + 1.2 * phi), abs_tol=1e-12):
                raise ValueError("input must use original boundary mapping")
            new = -(.5 + (unit_flux - .5) * phi)
        parameters = SafeSAGEGuidanceParameters(
            ClassSafetyParameters(phi, -new), row["lambda"], row.get("applied_passing_side", "right"))
        policies[key] = parameters
        audit.append(dict(track_id=key, phi=phi, original_boundary_flux=old,
                          candidate_boundary_flux=new, qualified=key in changed,
                          lambda_value=row["lambda"]))
    return policies, audit


def probe_configuration(footprint):
    """Mirror the archived restaurant recipe, independently of Isaac imports."""
    return SafeSAGEFieldPipelineConfig(
        SafeSAGEConfigurationSpaceConfig(footprint=footprint, yaw_slice_count=16,
                                         expansion_backend="vectorized"),
        TangentialLaplaceGuidanceFieldConfig(convergence_tolerance=1e-6,
            tangential_buffer_radius=.60, allow_dense_scene_arbitration=True,
            merge_equal_policy_regions=True, solver_backend="sparse_direct",
            interface_backend="vectorized"),
        SafeSAGEHFullConfig(yaw_slice_count=16, softplus_beta=1., minimum_forcing_magnitude=1e-9,
            poisson_config=PoissonSafetyFunctionConfig(convergence_tolerance=1e-6,
                                                       solver_backend="sparse_direct")))


def reconstruct_pipeline(record, geometry, anchor, footprint, unit_flux):
    """Rasterize exactly the archived placements/IDs; no oracle obstacle geometry."""
    bounds = record["diagnostics"]["rolling_local_field"]["current_bounds"]
    config = LocalOccupancyGridConfig(**{k: v for k, v in bounds.items() if not k.startswith("_")})
    points = np.concatenate((geometry["points_world_xy"], geometry["retained_world_xy"]))
    ids = np.concatenate((geometry["point_track_ids"], geometry["retained_track_ids"]))
    if len(points) != record["diagnostics"]["field_raw_placement_count"]:
        raise ValueError("placement count differs from native")
    risks = np.concatenate((geometry["point_phi"], geometry["retained_phi"]))
    policies, audit = mapped_policies(record, unit_flux)
    for key in np.unique(ids):
        if key not in policies or not np.allclose(risks[ids == key], policies[int(key)].risk, rtol=0, atol=1e-12):
            raise ValueError("archived placement policy mismatch")
    c, s = math.cos(anchor[2]), math.sin(anchor[2])
    points = (points - anchor[:2]) @ np.array([[c, -s], [s, c]])
    cells = np.floor((points - [config.minimum_x, config.minimum_y]) / config.resolution).astype(np.int64)
    _, first = np.unique(np.column_stack((cells, ids)), axis=0, return_index=True)
    first.sort()
    if len(first) != record["diagnostics"]["field_cell_representative_count"]:
        raise ValueError("cell representative count differs from native")
    unknown = ClassSafetyParameters(1., 1.7)
    source = build_local_occupancy_grid(points[first], point_track_ids=ids[first],
        class_safety_policy=ClassSafetyPolicy({"unknown": unknown}), config=config,
        track_safety_parameters={k: p.safety_parameters for k, p in policies.items()})
    result = build_safe_sage_field_pipeline(source, config=probe_configuration(footprint),
        guidance_policy=SafeSAGEGuidancePolicy({"unknown": SafeSAGEGuidanceParameters(unknown, 1.25, "right")}),
        track_guidance_parameters=policies)
    return result, audit


def pipeline_difference(first, second):
    """Preserve geometry and disclose all scalar/vector numerical differences."""
    same_geometry = all(np.array_equal(a.occupied_mask, b.occupied_mask)
        for a, b in zip(first.configuration_space.occupancy_grids, second.configuration_space.occupancy_grids))
    if not same_geometry:
        raise ValueError("boundary-only diagnostic changed occupied configuration space")
    return dict(geometry_identical=True,
        h_max_abs_difference=float(np.max(np.abs(first.h_full.values - second.h_full.values))),
        forcing_max_abs_difference=float(np.max(np.abs(first.h_full.forcing_fields - second.h_full.forcing_fields))),
        guidance_max_abs_difference=max(float(np.max(np.abs(a.vectors_robot_xy - b.vectors_robot_xy)))
            for a, b in zip(first.guidance_fields, second.guidance_fields)))


def static_analytical_probe(control, template_field, pipeline, state, command):
    """Same pose/command, zero temporal term: explicitly not native CBF replay."""
    field = replace(template_field, pipeline=pipeline, temporal_safety_derivative=0.0)
    result = control.filter_command(command, np.asarray(state), field)
    return dict(command=[result.safe_linear_velocity, result.safe_angular_velocity],
        cbf_margin=result.cbf_margin, constraint_feasible=result.constraint_feasible,
        constraint_satisfied=result.constraint_satisfied,
        emergency_stop_active=result.emergency_stop_active,
        sampled_h=result.sampled_safety_value, sampled_gradient=list(result.sampled_gradient),
        sampled_guidance=list(result.sampled_guidance_grid_xy),
        linear_velocity_coefficient=result.linear_velocity_coefficient,
        angular_velocity_coefficient=result.angular_velocity_coefficient,
        right_hand_side=result.constraint_right_hand_side,
        scope="frozen geometry with temporal derivative zero; not live temporal/MPC replay")


def cold_mpc_probe(control, pipeline, state, command):
    """N12 primary objective from saved limits, no guidance/warm-start/history."""
    from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPC
    mpc = SafeSAGEUnicycleMPC(control.mpc.paper_filter,
                            replace(control.mpc.config, horizon_steps=12))
    result = mpc.plan(state, command, configuration_space=pipeline.configuration_space,
        h_full=pipeline.h_full, guidance_fields=pipeline.guidance_fields,
        temporal_safety_derivative=0.)
    return dict(success=result.success, status=result.status, first_command=list(result.first_command),
        predicted_chassis_states=np.asarray(result.predicted_chassis_states).tolist(),
        minimum_dtcbf_margin=result.minimum_dtcbf_margin,
        scope="matched cold N12 primary objective from saved limits; zero temporal term; no guidance objective or native warm-start")
