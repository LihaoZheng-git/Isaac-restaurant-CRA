"""Stationary sphere evidence, matched CRA fields, then optional wheel pass-by.

Runtime geometry comes from real Pandar returns. Low/medium/high factor values
are separately labeled ORACLE experimental inputs, not outputs of the vision
estimators. All fields use identical geometry and lambda=1.25. Sensor evidence
keeps unavailable mass/contact/thermal inputs and the normal conservative CRA
policy. No restaurant asset is saved or enriched by this session-layer fixture.

Run stationary first, then explicitly request the controlled motion comparison:
    python -u scripts/demos/cra_simple_object_isaac_demo.py --output outputs/simple_object/stationary
    python -u scripts/demos/cra_simple_object_isaac_demo.py --motion --output outputs/simple_object/motion
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--motion", action="store_true")
parser.add_argument("--grid-resolution", type=float, default=0.10)
parser.add_argument("--yaw-slices", type=int, default=16)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WARP_CACHE_PATH", str(args.output / "warp"))

# Reuse the validated scene, footprint, PDE and controller configurations.
# This creates SimulationApp once; imported main functions do not execute.
from scripts.demos import safe_sage_unicycle_mpc_motion_demo as existing

import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import omni.timeline
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from robot.control.differential_drive import NovaCarterController
from robot.evaluation.factor_specific_cra import diagnose_factor_specific_cra
from robot.evaluation.cra_experiments import EvidenceMode, RestaurantScenario
from robot.evaluation.perception_readiness import evaluate_perception_readiness
from robot.evaluation.simple_object_test import (
    ControlledObjectInputs,
    controlled_object_adapter,
    disk_footprint_clearance,
    run_bounded_pass_by,
)
from robot.perception.fall_height import estimate_fall_height
from robot.perception.lidar_fall_surfaces import estimate_lidar_fall_surfaces
from robot.perception.motion_attributes import RouteMotionContext, estimate_track_motion
from robot.perception.physical_attributes import PhysicalAttributes
from robot.perception.physical_geometry import estimate_track_geometry
from robot.risk.physical_factor_adapter import adapt_physical_factors
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy
from robot.safety.class_safety_policy import ClassSafetyParameters
from robot.safety.local_occupancy_grid import build_local_occupancy_grid
from robot.safety.obstacle_detection import (
    ObstacleDetectionConfig,
    detect_obstacle_points_3d,
)
from robot.safety.obstacle_tracking import ObstacleTracker, ObstacleTrackerConfig
from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint
from robot.safety.safe_sage_field_pipeline import build_safe_sage_field_pipeline
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPC
from robot.safety.safe_sage_guidance_policy import (
    SafeSAGEGuidanceParameters,
    PassingSide,
)
from robot.safety.semantic_obstacle import SemanticObstacle
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import compute_relative_prim_transform
from robot.sensor.camera import NovaCarterCamera
from robot.sensor.lidar import NovaCarterLidar
from robot.sensor.lidar_geometry import transform_points

stationary = existing.stationary
simulation_app = existing.simulation_app
RADIUS_M = 0.25
CENTER_WORLD_XY = np.array(
    [
        stationary.ROBOT_START_POSITION_WORLD[0] + 1.30,
        stationary.ROBOT_START_POSITION_WORLD[1],
    ]
)
FIXTURE_PATH = "/World/CRASimpleObject/Sphere"
GOAL_XY = (2.50, 0.0)
PROFILE_INPUTS = {
    "controlled_low": ControlledObjectInputs(),
    "controlled_medium": ControlledObjectInputs(contact_proxy=0.40),
    "controlled_high": ControlledObjectInputs(contact_proxy=0.80),
}


def save_json(name, payload):
    (args.output / name).write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def local_state(stage, anchor):
    pose = stationary.read_robot_pose(stage)
    xy, yaw = existing.motion.current_pose_in_field_frame(pose, anchor_pose=anchor)
    return np.array([*xy, yaw])


def main():
    class_policy, guidance_policy, grid_config, field_config = (
        existing.create_fixed_field_configuration()
    )
    # Finer discretization avoids the coarse demo grid's false unsafe cells
    # during the sphere turn. The measured footprint/margin are unchanged.
    grid_config = replace(grid_config, resolution=args.grid_resolution)
    field_config = replace(
        field_config,
        configuration_space_config=replace(
            field_config.configuration_space_config, yaw_slice_count=args.yaw_slices
        ),
        h_full_config=replace(
            field_config.h_full_config, yaw_slice_count=args.yaw_slices
        ),
    )
    loaded = load_robot_scene(
        scene_path=stationary.ROBOT_SCENE_PATH,
        simulation_app=simulation_app,
        robot_prim_path=stationary.ROBOT_PRIM_PATH,
        expected_meters_per_unit=stationary.EXPECTED_METERS_PER_UNIT,
    )
    stage = loaded.stage
    root_before = stage.GetRootLayer().ExportToString()
    scene_hash_before = hashlib.sha256(
        Path(stationary.ROBOT_SCENE_PATH).read_bytes()
    ).hexdigest()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        sphere = UsdGeom.Sphere.Define(stage, FIXTURE_PATH)
        sphere.CreateRadiusAttr(RADIUS_M)
        sphere.AddTranslateOp().Set(Gf.Vec3d(*CENTER_WORLD_XY, RADIUS_M))
        sphere.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.45, 0.95)])
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    root_unchanged_by_fixture = stage.GetRootLayer().ExportToString() == root_before
    controller = NovaCarterController(robot_prim_path=stationary.ROBOT_PRIM_PATH)
    SimulationManager.setup_simulation(
        dt=stationary.PHYSICS_DT, device=stationary.PHYSICS_DEVICE
    )
    omni.timeline.get_timeline_interface().play()
    app_utils.update_app(steps=10)
    controller.verify_articulation()
    stationary.position_robot_for_experiment(controller)
    for _ in range(stationary.SETTLE_STEPS):
        controller.stop()
        simulation_app.update()
    anchor = stationary.read_robot_pose(stage)
    robot_from_lidar = compute_relative_prim_transform(
        stage,
        source_prim_path=stationary.PANDAR_XT32_LIDAR_PRIM_PATH,
        target_prim_path=stationary.ROBOT_BASE_PRIM_PATH,
    )
    floor = stationary.estimate_floor_height_in_robot_frame(stage)
    detection = ObstacleDetectionConfig(
        minimum_height=floor + 0.05,
        maximum_height=floor + 1.0,
        minimum_range=0.15,
        maximum_range=5.0,
        forward_only=False,
        angular_bin_size_degrees=None,
    )
    lidar = NovaCarterLidar(lidar_prim_path=stationary.PANDAR_XT32_LIDAR_PRIM_PATH)
    lidar.initialize()
    camera = NovaCarterCamera()
    camera.initialize()
    tracker = ObstacleTracker(
        ObstacleTrackerConfig(
            cluster_radius=0.20,
            minimum_cluster_points=4,
            maximum_association_distance=0.50,
        )
    )
    last_timestamp = None
    samples = []
    for _ in range(400):
        controller.stop()
        simulation_app.update()
        try:
            raw = lidar.get_point_cloud()
            info = lidar.get_frame_info()
        except RuntimeError:
            continue
        if info.sensor_timestamp_nanoseconds == last_timestamp:
            continue
        last_timestamp = info.sensor_timestamp_nanoseconds
        scene_xyz = transform_points(raw, robot_from_lidar)
        obstacle_xyz = detect_obstacle_points_3d(scene_xyz, detection)
        pose = stationary.read_robot_pose(stage)
        tracking = tracker.update(
            obstacle_xyz[:, :2],
            robot_position_world_xy=np.array([pose.x, pose.y]),
            robot_yaw=pose.yaw,
            timestamp_seconds=last_timestamp * 1e-9,
        )
        target, error = existing.motion.find_controlled_proxy_track_world(
            tracking.tracked_obstacles, CENTER_WORLD_XY
        )
        if target is None or error > 0.40:
            continue
        samples.append((scene_xyz, obstacle_xyz, tracking, target, info, pose, error))
        if len(samples) == 12:
            break
    if len(samples) < 12:
        raise RuntimeError(
            f"sphere capture insufficient: {len(samples)} matched frames"
        )
    scene_xyz, obstacle_xyz, tracking, target, info, final_pose, association_error = (
        samples[-1]
    )
    timestamp = info.sensor_timestamp_nanoseconds * 1e-9
    selected = tracking.point_track_ids == target.track_id
    object_xyz = obstacle_xyz[selected]
    points_anchor = existing.motion.robot_points_in_anchor_frame(
        object_xyz[:, :2], robot_pose=final_pose, anchor_pose=anchor
    )
    geometry = estimate_track_geometry(
        obstacle_xyz,
        tracking_result=tracking,
        track_id=target.track_id,
        floor_height_robot_m=floor,
    )
    surfaces = estimate_lidar_fall_surfaces(
        scene_xyz, object_xyz, timestamp_seconds=timestamp
    )
    if surfaces.is_available:
        geometry = replace(
            geometry,
            fall_height_m=estimate_fall_height(
                surfaces.evidence, timestamp_seconds=timestamp
            ),
        )
    motion = estimate_track_motion(
        target,
        timestamp_seconds=timestamp,
        route_context=RouteMotionContext(
            (final_pose.x, final_pose.y), tuple(CENTER_WORLD_XY + [1.2, 0]), 0.20
        ),
    )
    base = PhysicalAttributes.unknown(
        track_id=target.track_id,
        timestamp_seconds=timestamp,
        semantic_class="object",
        semantic_source="controlled_track_selection_prior",
        semantic_confidence=1.0,
    )
    sensor_attributes = replace(base, geometry=geometry, motion=motion)
    sensor_adapter = adapt_physical_factors(
        sensor_attributes, timestamp_seconds=timestamp
    )
    sensor_policy = CRASafeSAGEPolicy().evaluate(sensor_adapter.profile)
    np.savez_compressed(
        args.output / "capture.npz",
        scene_xyz=scene_xyz,
        object_xyz=object_xyz,
        points_anchor=points_anchor,
        timestamps=[
            sample[4].sensor_timestamp_nanoseconds * 1e-9 for sample in samples
        ],
        object_velocities=[sample[3].velocity_world_xy for sample in samples],
    )
    try:
        plt.imsave(args.output / "camera.png", camera.get_rgb())
    except RuntimeError as error:
        print(f"Camera visualization unavailable: {error}")
    drift = math.hypot(final_pose.x - anchor.x, final_pose.y - anchor.y)
    yaw_drift = abs(final_pose.yaw - anchor.yaw)
    stationary_passed = (
        drift <= 0.02
        and yaw_drift <= math.radians(1)
        and len(object_xyz) >= 4
        and association_error <= 0.40
        and root_unchanged_by_fixture
    )
    # Known position is only a controlled track-selection and independent
    # collision-scoring input; no authored radius/center enters the field.
    delta = CENTER_WORLD_XY - np.array([anchor.x, anchor.y])
    c, s = math.cos(anchor.yaw), math.sin(anchor.yaw)
    sphere_center = np.array(
        [c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]]
    )
    footprint = RectangularRobotFootprint(
        stationary.ROBOT_FRONT_EXTENT,
        stationary.ROBOT_REAR_EXTENT,
        stationary.ROBOT_LEFT_EXTENT,
        stationary.ROBOT_RIGHT_EXTENT,
        stationary.ROBOT_SAFETY_MARGIN,
    )
    semantic = SemanticObstacle(
        target, "object", 1.0, "controlled_track_selection_prior"
    )
    fields, policies, factors, predictions, readiness = {}, {}, {}, {}, {}
    mpc, backup = existing.create_controller_stack(controller)
    # Match the geometry checks to the actual physics-step CBF rechecks.
    # The old frozen-field demo only checked MPC interval endpoints.
    mpc = SafeSAGEUnicycleMPC(
        mpc.paper_filter,
        replace(
            mpc.config,
            safety_substeps_per_interval=round(
                mpc.config.time_step / stationary.PHYSICS_DT
            ),
            integration_substeps_per_interval=round(
                mpc.config.time_step / stationary.PHYSICS_DT
            ),
            enable_progress_seeds=True,
            progress_seed_linear_fraction=0.5,
            # Prefix envelopes across 1,056 preceding wheel intervals cover
            # lookahead-position/yaw errors, including command reversals.
            configuration_position_padding_m=0.025,
            configuration_yaw_padding_radians=0.21,
            configuration_padding_initial_fraction=0.1,
        ),
    )
    # Stationary initial predictions never command the wheels.
    from robot.navigation.waypoint_controller import (
        RobotPose2D,
        Waypoint,
        WaypointController,
        WaypointControllerConfig,
    )

    waypoint = WaypointController(
        [Waypoint(*GOAL_XY)],
        config=WaypointControllerConfig(
            maximum_linear_velocity=0.20,
            maximum_angular_velocity=0.60,
            rotate_in_place_threshold_radians=math.radians(100),
            minimum_alignment_scale=0.25,
        ),
    )
    for name, inputs in PROFILE_INPUTS.items():
        adapted = controlled_object_adapter(
            inputs, track_id=target.track_id, timestamp_seconds=timestamp
        )
        policy = CRASafeSAGEPolicy().evaluate(adapted.profile)
        readiness[name] = asdict(
            evaluate_perception_readiness(
                RestaurantScenario(
                    "simple_object",
                    "contact",
                    "controlled sphere test",
                    EvidenceMode.ORACLE,
                    0,
                ),
                adapted,
            )
        )
        grid = build_local_occupancy_grid(
            points_anchor,
            point_track_ids=np.full(len(points_anchor), target.track_id),
            semantic_obstacles=(semantic,),
            class_safety_policy=class_policy,
            config=grid_config,
            track_safety_parameters={
                target.track_id: policy.guidance_parameters.safety_parameters
            },
        )
        field = build_safe_sage_field_pipeline(
            grid,
            guidance_policy=guidance_policy,
            config=field_config,
            semantic_obstacles=(semantic,),
            track_guidance_parameters={target.track_id: policy.guidance_parameters},
        )
        fields[name] = field
        policies[name] = policy.as_dict()
        factors[name] = diagnose_factor_specific_cra(adapted).to_dict()
        np.savez_compressed(
            args.output / f"{name}-field.npz", h_full=field.h_full.values
        )
        nominal = waypoint.predict_commands(
            RobotPose2D(0.0, 0.0, 0.0),
            horizon_steps=mpc.config.horizon_steps,
            time_step=mpc.config.time_step,
        )
        planned = mpc.plan(
            np.zeros(3),
            np.array(
                [[item.linear_velocity, item.angular_velocity] for item in nominal]
            ),
            configuration_space=field.configuration_space,
            h_full=field.h_full,
            guidance_fields=field.guidance_fields,
            temporal_safety_derivative=0.0,
            preferred_lateral_direction_grid_y=-1.0,
        )
        predictions[name] = {
            "success": existing.predictive_result_is_valid(
                planned, constraint_tolerance=mpc.config.constraint_tolerance
            ),
            "status": planned.status,
            "minimum_margin": float(np.min(planned.dtcbf_margins)),
            "predicted_chassis_states": planned.predicted_chassis_states.tolist(),
        }
    # A constant policy baseline shares CRA's flux range and lambda, so a
    # different legacy flux range or class-specific lambda cannot confound it.
    fixed_risk = 0.40
    fixed_guidance = SafeSAGEGuidanceParameters(
        ClassSafetyParameters(
            fixed_risk,
            CRASafeSAGEPolicy().config.normal_flux_magnitude_for_risk(fixed_risk),
        ),
        1.25,
        PassingSide.RIGHT,
    )
    fixed_grid = build_local_occupancy_grid(
        points_anchor,
        point_track_ids=np.full(len(points_anchor), target.track_id),
        semantic_obstacles=(semantic,),
        class_safety_policy=class_policy,
        config=grid_config,
        track_safety_parameters={target.track_id: fixed_guidance.safety_parameters},
    )
    fields["fixed_parameter_baseline"] = build_safe_sage_field_pipeline(
        fixed_grid,
        guidance_policy=guidance_policy,
        config=field_config,
        semantic_obstacles=(semantic,),
        track_guidance_parameters={target.track_id: fixed_guidance},
    )
    policies["fixed_parameter_baseline"] = {
        "cra": {
            "phi": fixed_risk,
            "source": "constant experimental baseline, not physical assessment",
        },
        "safe_sage": {
            "boundary_flux_magnitude": fixed_guidance.boundary_flux_magnitude,
            "tangential_strength": 1.25,
        },
    }
    fixed_field = fields["fixed_parameter_baseline"]
    fixed_prediction = mpc.plan(
        np.zeros(3),
        np.array([[item.linear_velocity, item.angular_velocity] for item in nominal]),
        configuration_space=fixed_field.configuration_space,
        h_full=fixed_field.h_full,
        guidance_fields=fixed_field.guidance_fields,
        temporal_safety_derivative=0.0,
        preferred_lateral_direction_grid_y=-1.0,
    )
    predictions["fixed_parameter_baseline"] = {
        "success": existing.predictive_result_is_valid(
            fixed_prediction, constraint_tolerance=mpc.config.constraint_tolerance
        ),
        "status": fixed_prediction.status,
        "minimum_margin": float(np.min(fixed_prediction.dtcbf_margins)),
        "predicted_chassis_states": fixed_prediction.predicted_chassis_states.tolist(),
    }
    phi = [policies[name]["cra"]["phi"] for name in PROFILE_INPUTS]
    flux = [
        policies[name]["safe_sage"]["boundary_flux_magnitude"]
        for name in PROFILE_INPUTS
    ]
    mapping_passed = (
        all(a < b for a, b in zip(phi, phi[1:]))
        and all(a < b for a, b in zip(flux, flux[1:]))
        and all(np.isfinite(field.h_full.values).all() for field in fields.values())
        and all(
            item["safe_sage"]["tangential_strength"] == 1.25
            for item in policies.values()
        )
    )
    gate = (
        stationary_passed
        and mapping_passed
        and all(item["success"] for item in predictions.values())
        and all(item["ready"] for item in readiness.values())
    )
    report = {
        "stationary_passed": stationary_passed,
        "mapping_passed": mapping_passed,
        "motion_gate_passed": gate,
        "evidence_mode": "oracle profiles with measured LiDAR geometry",
        "aggregate_sensor_acceptance": False,
        "sensor_phi": sensor_policy.assessment.phi,
        "sensor_factors": diagnose_factor_specific_cra(sensor_adapter).to_dict(),
        "sensor_attributes": asdict(sensor_attributes),
        "root_layer_unchanged_by_fixture": root_unchanged_by_fixture,
        "position_drift_m": drift,
        "yaw_drift_degrees": math.degrees(yaw_drift),
        "target_point_count": len(object_xyz),
        "target_track_id": target.track_id,
        "association_error_m": association_error,
        "distinct_matched_frames": len(samples),
        "fall_surface_rejections": surfaces.rejection_reasons,
        "profiles": {name: asdict(value) for name, value in PROFILE_INPUTS.items()},
        "policies": policies,
        "factor_diagnostics": factors,
        "stationary_predictions": predictions,
        "controlled_readiness": readiness,
        "fixed_lambda": 1.25,
        "grid_configuration": asdict(grid_config),
        "field_configuration": asdict(field_config),
        "mpc_configuration": asdict(mpc.config),
        "capture_sha256": hashlib.sha256(
            (args.output / "capture.npz").read_bytes()
        ).hexdigest(),
        "assumptions": [
            "frozen field and CRA assessment during each pass-by; static sphere; h_t=0",
            "robot ego pose and LiDAR extrinsics are simulation-oracle USD references; world-Z-zero floor is a configured prior",
            "controlled factor values are illustrative oracle inputs, not calibrated model predictions",
            "contact proxy low value is supplied; absent edge detection is not negative evidence",
            "thermal low value is supplied; no plume is not proof of a cold object",
            "kinetic uses nonzero 0.20 m/s relative approach speed even though the object is stationary",
            "authored location selects the target track; authored radius only scores independent clearance",
            "mass/contact/thermal for the sphere remain unavailable in the sensor record",
            "fixed geometry, waypoint, controller limits and lambda across comparisons",
        ],
        "motion": {},
    }
    save_json("stationary.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "stationary_passed",
                    "mapping_passed",
                    "motion_gate_passed",
                    "sensor_phi",
                    "target_point_count",
                    "position_drift_m",
                )
            }
        ),
        flush=True,
    )
    if args.motion and not gate:
        raise RuntimeError("stationary gate rejected controlled wheel motion")
    if args.motion:
        initial_dof_positions = controller.robot.get_dof_positions().numpy().copy()
        for name in ("controlled_low", "fixed_parameter_baseline", "controlled_high"):
            controller.stop()
            controller.robot.set_velocities(
                linear_velocities=[[0.0, 0.0, 0.0]],
                angular_velocities=[[0.0, 0.0, 0.0]],
            )
            controller.robot.set_dof_velocities(np.zeros_like(initial_dof_positions))
            controller.robot.set_dof_positions(initial_dof_positions)
            stationary.position_robot_for_experiment(controller)
            for _ in range(stationary.SETTLE_STEPS):
                controller.stop()
                simulation_app.update()
            state_before = local_state(stage, anchor)
            if np.linalg.norm(state_before[:2]) > 0.02 or abs(
                state_before[2]
            ) > math.radians(1):
                raise RuntimeError(
                    f"matched trial reset failed: {state_before.tolist()}"
                )

            def advance(v, omega):
                if not simulation_app.is_running() or not app_utils.is_playing():
                    raise RuntimeError("simulation stopped during bounded motion")
                controller.apply_command(v, omega)
                simulation_app.update()

            result = run_bounded_pass_by(
                field=fields[name],
                mpc=mpc,
                backup=backup,
                read_state=lambda: local_state(stage, anchor),
                advance=advance,
                stop=controller.stop,
                clearance=lambda state: disk_footprint_clearance(
                    state,
                    center_xy=sphere_center,
                    radius_m=RADIUS_M,
                    footprint=footprint,
                ),
                goal_xy=GOAL_XY,
                physics_dt=stationary.PHYSICS_DT,
                progress=lambda row: print(
                    json.dumps({"profile": name, "progress": row}), flush=True
                ),
            )
            result["motion_mode"] = "Isaac Nova Carter physical wheel commands"
            result["initial_state"] = state_before.tolist()
            stopping_states = []
            stopping_clearances = []
            for _ in range(30):
                advance(0.0, 0.0)
                state = local_state(stage, anchor)
                stopping_states.append(state.tolist())
                stopping_clearances.append(disk_footprint_clearance(
                    state, center_xy=sphere_center, radius_m=RADIUS_M,
                    footprint=footprint,
                ))
            stopping_passed = bool(
                np.isfinite(stopping_states).all()
                and min(stopping_clearances) > 0.02
            )
            result["stopping"] = {
                "simulated_seconds": 30 * stationary.PHYSICS_DT,
                "minimum_clearance_m": min(stopping_clearances),
                "trajectory": stopping_states,
                "passed": bool(stopping_passed),
            }
            result["passed"] = result["passed"] and stopping_passed
            report["motion"][name] = result
            save_json("report.json", report)
            print(
                json.dumps(
                    {
                        "profile": name,
                        **{
                            key: value
                            for key, value in result.items()
                            if key not in {"trajectory", "commands", "nominal_commands"}
                        },
                    }
                ),
                flush=True,
            )
    report["passed"] = gate and (
        not args.motion or all(item["passed"] for item in report["motion"].values())
    )
    report["scene_file_unchanged"] = (
        hashlib.sha256(Path(stationary.ROBOT_SCENE_PATH).read_bytes()).hexdigest()
        == scene_hash_before
    )
    report["passed"] = report["passed"] and report["scene_file_unchanged"]
    save_json("report.json", report)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(
        points_anchor[:, 0],
        points_anchor[:, 1],
        s=8,
        c="black",
        label="Pandar target returns",
    )
    axis.add_patch(
        plt.Circle(
            sphere_center,
            RADIUS_M,
            color="steelblue",
            alpha=0.3,
            label="oracle scoring outline",
        )
    )
    axis.plot([0, GOAL_XY[0]], [0, GOAL_XY[1]], "k--", label="nominal waypoint line")
    for name, result in report["motion"].items():
        states = np.asarray(result["trajectory"])
        axis.plot(
            states[:, 0],
            states[:, 1],
            label=f"{name}, phi={policies[name]['cra']['phi']:.3f}",
        )
    if not report["motion"]:
        for name, result in predictions.items():
            states = np.asarray(result["predicted_chassis_states"])
            axis.plot(states[:, 0], states[:, 1], label=f"{name} stationary prediction")
    axis.scatter(*GOAL_XY, marker="*", s=100, c="green", label="goal")
    axis.set(
        xlabel="Trial-frame x (m)",
        ylabel="Trial-frame y (m)",
        title="Simple object: controlled CRA, measured geometry, lambda=1.25",
        xlim=(-0.3, 2.8),
        ylim=(-1.3, 1.0),
    )
    axis.set_aspect("equal")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "paths.png", dpi=160)
    plt.close(fig)
    controller.stop()
    omni.timeline.get_timeline_interface().stop()
    if not report["passed"]:
        raise RuntimeError(
            "simple-object test did not meet every acceptance criterion; see report.json"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        save_json(
            "failure.json",
            {
                "error": str(error),
                "traceback": traceback.format_exc(),
                "motion_requested": args.motion,
            },
        )
        raise
    finally:
        simulation_app.close()
