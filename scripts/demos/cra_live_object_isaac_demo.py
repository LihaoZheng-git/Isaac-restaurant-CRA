"""Live RGB/LiDAR CRA in a bounded session fixture, stationary gate first.

Perception and MPC run off the simulation thread on copied observations.
During navigation the simulator advances continuously; stale evidence/failed
control causes a recorded safety stop, never a scheduled observation pause.
Simulation pacing is explicit and is not a real-time performance claim.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--motion", action="store_true")
parser.add_argument("--experiment-baseline", choices=("geometric_velocity_cbf", "fixed_class_safe_sage", "continuous_cra_safe_sage"),
                    default="continuous_cra_safe_sage", help="explicit matched restaurant comparison; physical assessments always retained")
parser.add_argument("--motion-aware-passing", action="store_true",
                    help="opt-in current LiDAR crossing-side preference; unchanged CRA scores/lambda/safety checks")
parser.add_argument("--guidance-tracking-weight", type=float, default=0.0,
                    help="separate checked Laplace local-path MPC ablation; zero preserves primary")
parser.add_argument("--guidance-activation-h", type=float, default=1.0,
                    help="ablation reference activates smoothly below this safety-field value")
parser.add_argument("--guidance-max-heading-bias-degrees", type=float, default=45.0,
                    help="maximum local-reference heading bias about the active waypoint bearing")
parser.add_argument("--guidance-reference-goal-radius", type=float, default=0.04,
                    help="local reference stops inside this radius; actual waypoint acceptance is unchanged")
parser.add_argument("--guidance-fade-distance", type=float, default=0.5,
                    help="fade local guidance bias toward the active waypoint inside this distance")
parser.add_argument("--guidance-minimum-reference-progress", type=float, default=.01,
                    help="minimum checked-reference progress in metres: displacement in filtered mode, goal gain in legacy mode; zero disables this quality gate")
parser.add_argument("--guidance-turn-first-minimum-cap", type=float, default=.4,
                    help="minimum turn-first reference pool cap in rad/s; .1 reproduces the legacy pool; never an executable lower bound")
parser.add_argument("--guidance-reference-mode", choices=("bounded_field_heading", "filtered_goal_velocity", "planar_forecast_velocity"),
                    default="bounded_field_heading", help="explicit reference ablation; filtered goal velocity uses v_sem as the CBF normal")
parser.add_argument("--human-positive-fragment-extent", action="store_true",
                    help="bare-human recent-coobservation/current-positive-return fragment association; no free-space or factor transfer")
parser.add_argument("--human-observed-passing-target", action="store_true",
                    help="require fresh person/SAM association for new human passing choices; all obstacles retain CRA/safety intervention")
parser.add_argument("--human-passing-identity-prior-max-age", type=float, default=0.,
                    help="separate selection ablation: <=.5 s original person identity with current verified LiDAR envelope; zero keeps fresh-only")
parser.add_argument("--profile-runtime", action="store_true",
                    help="diagnostic worker call profiles for jobs over 3 wall seconds; profiling timings are not performance acceptance")
parser.add_argument("--save-planner-replay", action="store_true",
                    help="trusted local offline numeric snapshots, latest request per five-second source-time bucket; export counts against worker age")
parser.add_argument("--human-body-contact-prior", type=float,
                    help="opt-in fresh-person blunt-body contact PRIOR, e.g. .10; not measured sharpness")
parser.add_argument("--human-body-envelope-radius", type=float,
                    help="opt-in configured body-radius occupancy PRIOR; no actor geometry is read")
parser.add_argument("--human-envelope-centroid-uncertainty", type=float, default=0.2)
parser.add_argument("--human-envelope-identity-max-age", type=float, default=30.0)
parser.add_argument("--restaurant-scene", type=Path,
                    help="separate dressed restaurant scene; bounded 2.2 m real-bowl component route")
parser.add_argument("--restaurant-layout-survey", type=Path,
                    help="fingerprinted authored layout raster, independent clearance monitor only")
parser.add_argument("--restaurant-view", choices=("bowl", "pot", "counter", "front", "server"), default="bowl",
                    help="explicit stationary robot view; target names are evaluation metadata only")
parser.add_argument("--restaurant-route", choices=("short", "service", "human-service"), default="short")
parser.add_argument("--restaurant-server", action="store_true", help="explicit static-server scene opt-in; stationary actor gate required before wheels")
parser.add_argument("--restaurant-human", choices=("standing", "crossing", "continuous-crossing"),
                    help="native adult-human control integration; no static occupancy or shape persistence")
parser.add_argument("--human-camera-upward-degrees", type=float, default=0.,
                    help="explicit experimental left RGB mount pitch, 0..25 degrees; actual extrinsics re-read")
parser.add_argument("--human-camera-raise-metres", type=float, default=0.,
                    help="explicit experimental RGB mount height increase, 0...6 m")
parser.add_argument("--human-crossing-trigger-y", type=float, default=7.8,
                    help="explicit native actor-only trigger, world robot y 6.8..7.8; not a control/evidence input")
parser.add_argument("--human-crossing-delay-seconds", type=float,
                    help="opt-in exogenous actor start after navigation begins; shared timing across baselines, independent of robot progress")
parser.add_argument("--human-crossing-direction", choices=("right-to-left", "left-to-right"), default="right-to-left",
                    help="authored standalone actor fixture only; never supplies estimator velocity or passing direction")
parser.add_argument("--human-walking-speed", type=float, default=.7,
                    help="explicit native actor speed .5..1 m/s; never estimator input")
parser.add_argument("--human-cluster-radius", type=float, default=.12,
                    help="explicit human 3D connectivity radius .12...20 m; original occupancy and safety gates retained")
parser.add_argument("--human-carry-bowl", action="store_true",
                    help="native held bowl in explicit continuous-crossing mode; separate bowl perception remains unvalidated")
parser.add_argument("--human-upper-continuation", action="store_true",
                    help="cross once then continue in upper aisle; authored actor route only, no factor/control input")
parser.add_argument("--human-carry-facing-crossing", action="store_true",
                    help="explicit initial native actor heading along the first crossing leg; never evidence input")
parser.add_argument("--human-start-view", choices=("original", "approach-forward", "crossing-approach"), default="original",
                    help="explicit bounded carrying start; fixed world human-service waypoints remain unchanged")
parser.add_argument("--human-mass-prior", nargs=3, type=float, default=(70.,45.,100.),
                    metavar=("KG", "LOWER", "UPPER"), help="configured adult demonstration mass interval; not measured weight")
parser.add_argument("--sam-observed-context-box", action="store_true",
                    help="explicit generic observed-LiDAR context-box retry; unchanged SAM quality/ownership gates")
parser.add_argument("--restaurant-plume-diagnostic", action="store_true",
                    help="zero-wheel clean-reference / no-plume / visible-puff continuous cue evaluation")
parser.add_argument(
    "--persistent-object-risk-map",
    action="store_true",
    help="all-track numeric CRA with explicitly labeled factor/property priors",
)
parser.add_argument("--object-factor-prior-risk", type=float, default=0.25)
parser.add_argument("--fixture-layout", choices=("single", "three"), default="single")
parser.add_argument("--fixture-spacing", choices=("tight", "separated"), default="tight")
parser.add_argument("--progress-timeout-seconds", type=float, default=0.0)
parser.add_argument("--stationary-frames", type=int, default=12)
parser.add_argument("--kinetic-diagnostic", action="store_true")
parser.add_argument("--moving-actor", action="store_true")
parser.add_argument("--actor-shape", choices=("sphere", "upright_capsule"), default="sphere")
parser.add_argument("--visual-reassessment-interval", type=float, default=1.0,
                    help="explicit frozen-property vision retry interval; geometry/motion and numeric scores remain current")
parser.add_argument("--live-visual-tracks", type=int, choices=(1, 2), default=2,
                    help="existing fair visual-attempt budget per live update; all tracks still receive geometry/motion and numeric policies")
parser.add_argument("--vectorized-fields", action="store_true")
parser.add_argument("--object-height-evidence", action="store_true",
                    help="opt-in general LiDAR floor/elevation/support evidence; no carried-object relationship or hidden-base inference")
parser.add_argument("--height-support-margin", type=float, default=.12,
                    help="explicit general-height support neighborhood in metres; legacy defaults remain unchanged")
parser.add_argument("--height-support-gap", type=float, default=.06,
                    help="maximum visible-lower-boundary to support gap for general-height inference; not measured contact")
parser.add_argument("--stream-live-records", action="store_true",
                    help="append complete field records as compact JSONL; JSON checkpoint contains only the latest five fields")
parser.add_argument("--immediate-replan", action="store_true",
                    help="request a new solve without the periodic wait after rejected/invalidated plans; all execution gates remain")
parser.add_argument("--qp-iteration-limit", type=int, default=4000,
                    help="explicit live OSQP iteration budget; failed solves still require exact-safe recovery or stopping")
parser.add_argument("--nonlinear-rescue-iterations", type=int, default=300,
                    help="explicit optional SLSQP rescue budget; exact verification and recovery remain mandatory")
parser.add_argument("--progress-seed-fraction", type=float, default=.5,
                    help="search existing exact-safe turn seeds when MPC forward speed falls below this fraction of nominal; legacy .5, audited early-search .9")
parser.add_argument("--plume-view-fov", type=float, default=140.0)
parser.add_argument(
    "--evidence-gap-diagnostic",
    action="store_true",
    help="stationary-only controlled SAM-output loss and recovery; never a natural-occlusion claim",
)
parser.add_argument("--disable-evidence-retention", action="store_true")
parser.add_argument(
    "--disable-lidar-deskew",
    action="store_true",
    help="legacy whole-scan start-time transform baseline",
)
parser.add_argument(
    "--save-sensor-snapshots",
    action="store_true",
    help="save bounded early/periodic copied RGB/XYZ/timing diagnostic snapshots",
)
parser.add_argument(
    "--static-occupancy-memory-prior",
    action="store_true",
    help="explicit static fixture/environment assumption to retain positive occupancy through LiDAR blind regions",
)
parser.add_argument("--plume-control", choices=("none", "persistent"), default="none")
parser.add_argument(
    "--simulation-rate",
    type=float,
    default=0.1,
    help="maximum simulation seconds per wall second; reported explicitly",
)
parser.add_argument("--maximum-seconds", type=float, default=40.0)
parser.add_argument(
    "--planner-result-max-age",
    type=float,
    default=0.2,
    help="optimizer candidate age cap; every executed horizon still needs fresh exact validation",
)
args = parser.parse_args()
if (not math.isfinite(args.guidance_tracking_weight) or args.guidance_tracking_weight < 0
    or not math.isfinite(args.guidance_activation_h) or args.guidance_activation_h <= 0):
    parser.error("guidance tracking weight must be finite/nonnegative; activation h finite/positive")
if args.guidance_tracking_weight > 0 and args.experiment_baseline == "geometric_velocity_cbf":
    parser.error("guidance tracking is a separate MPC ablation, unavailable for geometric CBF")
if not math.isfinite(args.guidance_max_heading_bias_degrees) or not 0 < args.guidance_max_heading_bias_degrees < 90:
    parser.error("guidance heading bias must be finite and strictly between 0 and 90 degrees")
if not math.isfinite(args.guidance_reference_goal_radius) or args.guidance_reference_goal_radius <= 0:
    parser.error("guidance reference goal radius must be finite and positive")
if not math.isfinite(args.guidance_fade_distance) or args.guidance_fade_distance <= args.guidance_reference_goal_radius:
    parser.error("guidance fade distance must exceed the positive reference goal radius")
if not math.isfinite(args.guidance_minimum_reference_progress) or args.guidance_minimum_reference_progress < 0:
    parser.error("minimum guidance reference progress must be finite and nonnegative")
if not math.isfinite(args.guidance_turn_first_minimum_cap) or not 0 < args.guidance_turn_first_minimum_cap <= .8:
    parser.error("turn-first reference pool cap must be finite and in (0,.8]")
if args.human_observed_passing_target and not (args.motion_aware_passing and args.restaurant_human and not args.human_carry_bowl):
    parser.error("observed-human passing requires the bare-human motion-aware component")
if (not math.isfinite(args.human_passing_identity_prior_max_age)
    or not 0 <= args.human_passing_identity_prior_max_age <= .5
    or args.human_passing_identity_prior_max_age > 0 and (
        not args.human_observed_passing_target or args.human_body_envelope_radius is None)):
    parser.error("identity-prior passing requires observed-human/envelope mode and age in [0,.5]")
if args.human_body_envelope_radius is not None:
    from robot.perception.human_body_envelope import HumanBodyEnvelopeConfig
    try:
        HumanBodyEnvelopeConfig(body_radius_m=args.human_body_envelope_radius,
            centroid_uncertainty_m=args.human_envelope_centroid_uncertainty,
            maximum_identity_age_seconds=args.human_envelope_identity_max_age)
    except ValueError as exc:
        parser.error(str(exc))
    if not (args.restaurant_human and args.human_mass_prior and args.persistent_object_risk_map) or args.human_carry_bowl or args.static_occupancy_memory_prior:
        parser.error("human envelope requires standalone live human/person-prior object-map mode without static memory")
if args.human_positive_fragment_extent and (args.restaurant_human is None or args.human_carry_bowl):
    parser.error("positive fragment extent requires the bounded bare native-human scenario")
if not math.isfinite(args.progress_seed_fraction) or not 0 < args.progress_seed_fraction < 1:
    parser.error("progress seed fraction must be finite and in (0,1)")
if args.human_crossing_direction == "left-to-right" and args.restaurant_human != "crossing":
    parser.error("mirrored actor direction requires standalone crossing mode")
if args.motion_aware_passing and not (args.restaurant_human == "crossing" and args.persistent_object_risk_map
                                    and args.experiment_baseline != "geometric_velocity_cbf"):
    parser.error("bounded motion-aware passing requires the standalone crossing object-map scenario")
if args.human_crossing_delay_seconds is not None and (
    args.restaurant_human not in {"crossing", "continuous-crossing"}
    or not math.isfinite(args.human_crossing_delay_seconds)
    or not 0 <= args.human_crossing_delay_seconds <= 30
):
    parser.error("exogenous crossing delay requires crossing mode and 0..30 seconds")
if args.experiment_baseline != "continuous_cra_safe_sage" and not (args.restaurant_scene and args.persistent_object_risk_map):
    parser.error("matched baselines require the explicit persistent restaurant runner")
if args.human_body_contact_prior is not None and (
    not args.restaurant_human or not math.isfinite(args.human_body_contact_prior)
    or not 0 < args.human_body_contact_prior < 1
):
    parser.error("body contact PRIOR requires a native human and a finite score strictly between 0 and 1")
if (not math.isfinite(args.height_support_margin) or not .12 <= args.height_support_margin <= 1.
        or not math.isfinite(args.height_support_gap) or not .02 <= args.height_support_gap <= .20):
    raise ValueError("height support margin/gap must lie within [.12,1.] / [.02,.20] m")
if not args.object_height_evidence and (args.height_support_margin != .12 or args.height_support_gap != .06):
    raise ValueError("height support configuration requires --object-height-evidence")
if not .12 <= args.human_cluster_radius <= .20 or (args.human_cluster_radius != .12 and not args.restaurant_human):
    parser.error("human 3D radius .12...20 requires a native human scenario")
if not .5 <= args.human_walking_speed <= 1. or (
    args.human_walking_speed != .7 and args.restaurant_human not in {"crossing", "continuous-crossing"}
):
    parser.error("coupled walking-speed override requires crossing mode, .5..1 m/s")
if not 6.8 <= args.human_crossing_trigger_y <= 7.8 or (
    args.human_crossing_trigger_y != 7.8 and args.restaurant_human not in {"crossing", "continuous-crossing"}
):
    parser.error("crossing trigger override requires crossing mode, world y=6.8..7.8")
if args.restaurant_human == "continuous-crossing" and args.human_walking_speed != .5:
    parser.error("bounded continuous crossing currently requires .5 m/s")
if args.human_carry_bowl and args.restaurant_human != "continuous-crossing":
    parser.error("native carrying requires explicit continuous crossing")
if args.human_upper_continuation and not args.human_carry_bowl:
    parser.error("upper continuation requires the explicit carried-server scenario")
if args.human_carry_facing_crossing and not args.human_carry_bowl:
    parser.error("initial carrying heading requires the carried bowl")
if args.human_start_view != "original" and (not args.human_carry_bowl or args.restaurant_route != "human-service"):
    parser.error("forward robot start requires carrying and the human-service world route")
if not 0 <= args.human_camera_upward_degrees <= 25 or not 0 <= args.human_camera_raise_metres <= .6 or (
    (args.human_camera_upward_degrees or args.human_camera_raise_metres) and not args.restaurant_human
):
    parser.error("experimental RGB pitch 0..25 requires the native human mode")
if args.restaurant_scene and (
    not args.persistent_object_risk_map
    or (args.static_occupancy_memory_prior == bool(args.restaurant_human))
    or args.fixture_layout != "single" or args.actor_shape != "sphere"
    or args.plume_control != "none" or args.moving_actor or args.kinetic_diagnostic
    or args.evidence_gap_diagnostic or args.restaurant_layout_survey is None
):
    parser.error("restaurant component requires static persistent mapping, single/default selectors, no emitted plume/actors, and an independent authored layout survey")
if args.restaurant_human and (not args.restaurant_scene or not args.restaurant_server
    or args.restaurant_view != "bowl" or args.restaurant_plume_diagnostic
    or args.disable_lidar_deskew or args.restaurant_route not in {"service", "human-service"}):
    parser.error("native human requires restaurant service route/server, deskew, bowl start and no plume")
if args.restaurant_route == "human-service" and not args.restaurant_human:
    parser.error("human-service is the separate native-human centerline route")
if args.restaurant_layout_survey and not args.restaurant_scene:
    parser.error("restaurant layout survey requires the explicit restaurant scene")
if args.restaurant_view != "bowl" and (not args.restaurant_scene or args.motion):
    parser.error("additional restaurant views are stationary-only component diagnostics")
if args.restaurant_route != "short" and (not args.restaurant_scene or args.restaurant_view != "bowl"):
    parser.error("restaurant service route requires the validated bowl start view")
if args.restaurant_plume_diagnostic and (
    not args.restaurant_scene or args.restaurant_view != "bowl" or args.motion
    or not 12 <= args.stationary_frames <= 20
):
    parser.error("restaurant plume diagnostic requires zero wheels, bowl view and 12..20 frames per phase")
if args.restaurant_server and (not args.restaurant_scene or args.restaurant_plume_diagnostic):
    parser.error("static restaurant server requires a separate restaurant/no-plume diagnostic")
if args.restaurant_view == "server" and not args.restaurant_server:
    parser.error("server observation requires the explicit static-server scene opt-in")
if not 30 <= args.plume_view_fov < 170:
    parser.error("plume view FOV must lie within [30, 170) degrees")
if args.progress_timeout_seconds != 0 and not 10 <= args.progress_timeout_seconds <= 30:
    parser.error("progress watchdog must be disabled or within [10, 30] seconds")
if args.fixture_spacing != "tight" and args.fixture_layout != "three":
    parser.error("spacing option requires the three-object fixture")
if not (0.2 if args.restaurant_human else 1) <= args.visual_reassessment_interval <= 10:
    parser.error("visual reassessment interval must lie within [.2, 10] for native humans, [1, 10] otherwise")
if not 100 <= args.qp_iteration_limit <= 4000:
    parser.error("live QP iteration budget must lie within [100, 4000]")
if not 1 <= args.nonlinear_rescue_iterations <= 300:
    parser.error("nonlinear rescue budget must lie within [1, 300]")
if args.moving_actor and (not args.motion or args.kinetic_diagnostic or args.fixture_layout != "single" or not args.persistent_object_risk_map or args.static_occupancy_memory_prior or args.plume_control != "none"):
    parser.error("moving actor requires single/no-plume live motion, persistent map, and no static occupancy memory")
if args.actor_shape != "sphere" and (args.fixture_layout != "single" or args.plume_control != "none" or not args.persistent_object_risk_map or args.kinetic_diagnostic or args.evidence_gap_diagnostic):
    parser.error("capsule is a separate single/no-plume prior-aware kinetic fixture")
if args.kinetic_diagnostic and (args.motion or args.fixture_layout != "single" or not args.persistent_object_risk_map or args.static_occupancy_memory_prior or args.plume_control != "none"):
    parser.error("kinetic diagnostic requires single/no-plume persistent map, zero wheels, and no static occupancy memory")
if not 12 <= args.stationary_frames <= 60:
    parser.error("stationary frames must be within [12, 60]")
if args.fixture_layout != "single" and not args.persistent_object_risk_map:
    parser.error("multi-object fixture requires the all-track risk map")
if not 0 < args.object_factor_prior_risk < 1:
    parser.error("object factor prior must be strictly between zero and one")
if args.persistent_object_risk_map and args.evidence_gap_diagnostic:
    parser.error(
        "use cra_object_risk_map_replay.py for the new map's stationary gap diagnostic"
    )
if args.evidence_gap_diagnostic and (args.motion or args.disable_evidence_retention):
    parser.error("evidence-gap diagnostic requires retention and zero wheel motion")
if not 0 < args.simulation_rate <= 1 or not 0 < args.maximum_seconds <= 120:
    parser.error("invalid bounded simulation rate/duration")
if not 0.1 <= args.planner_result_max_age <= 0.5:
    parser.error("planner candidate age must be within [.1, .5] seconds")
args.output.mkdir(parents=True, exist_ok=True)
if args.stream_live_records and (args.output / "live-fields.jsonl").exists():
    parser.error("streaming diagnostics require a fresh output directory")
os.environ.setdefault("WARP_CACHE_PATH", str(args.output / "warp"))
os.environ.setdefault("ISAAC_SIM_HEADLESS", "1")
os.environ.setdefault("ISAAC_RTX_HISTORY_TRANSFORM_COUNT", "128")
if args.restaurant_human:
    os.environ["CRA_NATIVE_HUMAN"] = "1"

# Import the existing vision stack before Kit changes package resolution.
from robot.perception.visual_mass_runtime import (
    VisualMassRuntimeConfig,
    VisualMassRuntimeEstimator,
)
from robot.perception.sam_instance_segmentation import (
    SamInstanceConfig,
    SamInstanceResult,
    SamInstanceSegmenter,
)
from robot.perception.temporal_object_evidence import TemporalObjectEvidenceConfig
from robot.perception.persistent_object_evidence import PersistentObjectEvidenceConfig
from robot.risk.object_risk_policy import ObjectRiskPriorConfig
from robot.risk.cra import PhysicalRiskVector
from robot.perception.yolo_detector import (
    YoloSegmentationConfig,
    YoloSegmentationDetector,
)
from robot.perception.human_mass_prior import HumanMassPriorConfig, PERSON_CHECKPOINT_SHA256
from scripts.demos import safe_sage_unicycle_mpc_motion_demo as existing

import cv2
import carb
import numpy as np
import omni.timeline
from isaacsim.core.simulation_manager import SimulationManager
import isaacsim.core.experimental.utils.app as app_utils
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from robot.control.differential_drive import NovaCarterController
from robot.evaluation.live_object_perception import (
    LiveObjectPacket,
    LiveObjectPerception,
)
from robot.evaluation.restaurant_comparison import planner_result_diagnostic
from robot.evaluation.runtime_diagnostics import profile_call, numeric_runtime_contract
from robot.evaluation.factor_specific_cra import diagnose_factor_specific_cra
from robot.evaluation.kinetic_fixture import actor_offset_velocity, summarize_kinetic_records, summarize_live_relative_motion
from robot.evaluation.nearby_track_loss_guard import NearbyTrackLossConfig, NearbyTrackLossGuard
from robot.evaluation.object_map_validation import summarize_fixture_assessments, bounded_fixture_checks, kinetic_actor_fixture_checks
from robot.evaluation.restaurant_service_validation import (
    authored_raster_clearance, load_authored_layout_monitor, restaurant_asset_checks, restaurant_plume_checks,
)
from robot.evaluation.simple_object_test import disk_footprint_clearance, sample_stopping_sweep
from robot.navigation.waypoint_controller import (
    RobotPose2D,
    Waypoint,
    WaypointController,
    WaypointControllerConfig,
)
from robot.perception.live_inference import LiveInferenceRunner
from robot.perception.lidar_deskew import deskew_uncompensated_lidar
from robot.perception.physical_attributes import EstimateStatus
from robot.perception.lidar_fall_surfaces import LidarFallSurfaceConfig
from robot.perception.sensor_epochs import (
    EgoPoseHistory,
    EgoPoseSample,
    align_camera_lidar_epoch,
    gravity_aligned_robot_transform,
)
from robot.safety.obstacle_detection import (
    ObstacleDetectionConfig,
    detect_obstacle_points_3d,
)
from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint
from robot.safety.safe_sage_live_field import (
    SafeSAGELiveFieldBuffer,
    SafeSAGELiveFieldSnapshot,
)
from robot.safety.safe_sage_runtime import (
    SafeSAGEControlRuntime,
    SafeSAGEPlanExecutionLease,
)
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPC
from robot.safety.laplace_guidance_tracking_mpc import LaplaceGuidanceTrackingMPC, LaplaceGuidanceTrackingConfig
from robot.scene.camera_calibration import read_fisheye_polynomial_calibration
from robot.scene.loader import load_robot_scene
from robot.scene.transforms import compute_relative_prim_transform
from robot.scene.controlled_semantic_obstacle import read_world_bounds
from robot.scene.restaurant_service_build import (
    RESTAURANT_OBSERVATIONS, SERVICE_ROOT, restaurant_route_local_xy, service_props,
)
from robot.scene.restaurant_server import SERVER_OBSERVATION, SERVER_PRIM_PATH
from robot.sensor.camera import NovaCarterCamera
from robot.sensor.lidar import NovaCarterLidar
from robot.sensor.lidar_geometry import filter_valid_points, transform_points

stationary = existing.stationary
app = existing.simulation_app
ACTIVE_HUMAN = None
RADIUS = 0.18
ACTOR_HEIGHT = 1.4 if args.actor_shape == "upright_capsule" else 2 * RADIUS
ACTOR_CENTER_Z = ACTOR_HEIGHT / 2
GOAL = (3.8 if args.fixture_spacing == "separated" else 3.5 if args.fixture_layout == "three" else 2.5, 0.0)
SCENE_PATH = args.restaurant_scene.resolve() if args.restaurant_scene else Path(stationary.ROBOT_SCENE_PATH)
if args.restaurant_scene:
    restaurant_target, restaurant_start, restaurant_yaw = (SERVER_OBSERVATION
        if args.restaurant_view == "server" else RESTAURANT_OBSERVATIONS[args.restaurant_view])
    if args.human_start_view != "original":
        restaurant_yaw = math.pi / 2
    if args.human_start_view == "crossing-approach":
        restaurant_start = (28.8, 7.4, restaurant_start[2])
    stationary.ROBOT_START_POSITION_WORLD = restaurant_start
    stationary.ROBOT_START_YAW_RADIANS = restaurant_yaw
    GOAL = restaurant_route_local_xy(args.restaurant_route, human_start_view=args.human_start_view)[-1]
ROUTE = restaurant_route_local_xy(args.restaurant_route, human_start_view=args.human_start_view) if args.restaurant_scene else (GOAL,)
GROUND_SCOPE = "bounded separated ground sphere; observed flat plane with assumed unseen local support/landing continuity"
PLUME_SCOPE = "bounded rendered matte sphere and opaque bright puffs; validated initial clean reference; registered object-associated visible-plume proxy only"
if args.actor_shape == "upright_capsule":
    PLUME_SCOPE = "bounded separated matte upright capsule; validated initial clean reference; visible-plume proxy only, not body temperature"


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported report type: {type(value).__name__}")


def save_json(name, value):
    (args.output / name).write_text(
        json.dumps(value, indent=2, allow_nan=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def field_record(result, now, state=None):
    return {
        "sensor_frame_id": result.field.sensor_frame_id,
        "sensor_timestamp": result.field.sensor_timestamp_seconds,
        "commit_simulation_time": now,
        "build_wall_seconds": result.field.build_wall_seconds,
        "phi": result.policy.risk,
        "primary_record_scope": "first geometric track only; use objects for the all-track experiment"
        if result.objects
        else "legacy selected target",
        "objects": [
            {
                "track_id": item.attributes.track_id,
                "cra_control": item.policy.as_dict(),
                "readiness": asdict(item.readiness),
                "attributes": asdict(item.attributes),
                "diagnostics": item.diagnostics,
            }
            for item in result.objects
        ],
        "factor_specific_diagnostics": diagnose_factor_specific_cra(
            result.adapted
        ).to_dict(),
        "boundary_flux": result.policy.boundary_flux,
        "readiness": asdict(result.readiness),
        "factors": {
            name: asdict(getattr(result.adapted, name))
            for name in ("potential", "kinetic", "contact", "thermal")
        },
        "attributes": asdict(result.attributes),
        "diagnostics": result.diagnostics,
        "state_at_commit": None if state is None else state.tolist(),
    }


def save_instance_snapshot(result):
    if not args.save_sensor_snapshots or not result.detections.detections:
        return
    snapshot = args.output / "sensor-snapshots"
    snapshot.mkdir(exist_ok=True)
    np.savez_compressed(
        snapshot / f"{result.field.sensor_frame_id:06d}-instance.npz",
        mask=result.detections.detections[0].mask,
        camera_frame_id=result.detections.source_frame_id,
        camera_timestamp=result.detections.source_timestamp_seconds,
    )


def main():
    save_json("run-configuration.json", {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    })
    global ACTIVE_HUMAN
    print("Auditing/loading existing local models...", flush=True)
    weights = {}
    for name, variable, default in (
        ("sam", "SAM2_CHECKPOINT", "outputs/sam2_integration/sam2.1_t.pt"),
        (
            "mass",
            "VISUAL_MASS_CHECKPOINT",
            "outputs/visual_mass_data/abo/baseline_frozen_final_v2/best.pt",
        ),
        (
            "contact",
            "CONTACT_SEGMENTATION_CHECKPOINT",
            "runs/segment/outputs/contact_perception/training/yolo11n_seg_paco_geometry/weights/best.pt",
        ),
    ):
        path = Path(os.environ.get(variable, default)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{variable} requires existing local weights")
        weights[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    device = os.environ.get("YOLO_DEVICE", "0")
    segmenter = SamInstanceSegmenter(
        SamInstanceConfig(Path(weights["sam"]["path"]), device=device,
                          observed_context_box_retry=args.sam_observed_context_box,
                          foreground_depth_gap_m=0.5 if args.fixture_layout == "three" or args.restaurant_scene else None)
    )
    mass = VisualMassRuntimeEstimator(
        VisualMassRuntimeConfig(
            Path(weights["mass"]["path"]),
            device=device,
            allow_category_free_instances=True,
        )
    )
    contact = YoloSegmentationDetector(
        YoloSegmentationConfig(
            Path(weights["contact"]["path"]),
            device=device,
            confidence_threshold=0.25,
            allowed_classes=("exposed_edge",),
        )
    )
    contact.initialize()
    person = None
    if args.restaurant_human:
        path = ROOT / "outputs/restaurant_human_model_audit/yolo11n-seg.pt"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != PERSON_CHECKPOINT_SHA256:
            raise ValueError("person weights differ from the audited official checkpoint")
        weights["person"] = {"path": str(path), "sha256": digest}
        person = YoloSegmentationDetector(YoloSegmentationConfig(
            path, device=device, confidence_threshold=.5, allowed_classes=("person",)))
        person.initialize()
    print("Local models ready; loading the unchanged scene...", flush=True)
    original_hash = hashlib.sha256(
        SCENE_PATH.read_bytes()
    ).hexdigest()
    loaded = load_robot_scene(
        scene_path=SCENE_PATH,
        simulation_app=app,
        robot_prim_path=stationary.ROBOT_PRIM_PATH,
        expected_meters_per_unit=1.0,
    )
    stage = loaded.stage
    source_hashes = {layer.realPath: hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
        for layer in stage.GetLayerStack() if layer.realPath and Path(layer.realPath).is_file()}
    human = None
    if args.restaurant_human:
        from robot.scene.restaurant_human_episode import RestaurantHumanEpisode
        human = RestaurantHumanEpisode(stage, mode=args.restaurant_human, output=args.output,
                                       crossing_direction=args.human_crossing_direction,
                                       trigger_robot_y=args.human_crossing_trigger_y, speed_mps=args.human_walking_speed,
                                       crossing_delay_seconds=args.human_crossing_delay_seconds,
                                       carry_bowl=args.human_carry_bowl,
                                       carry_facing_crossing=args.human_carry_facing_crossing,
                                       upper_continuation=args.human_upper_continuation)
        ACTIVE_HUMAN = human
        save_json("human-native-setup.json", human.setup)
        for _ in range(20):
            app.update()
        human.bake()
    center_world = np.array(stationary.ROBOT_START_POSITION_WORLD[:2]) + [1.3, 0.0]
    fixture_specs = [
        {"name": "ground", "offset_xy": (1.3, 0.3 if args.fixture_layout == "three" else 0.0), "drop_m": 0.0,
         "color": (0.15, 0.45, 0.95), "plume": args.fixture_layout == "single" and args.actor_shape == "sphere"},
    ]
    if args.restaurant_scene:
        has_server = stage.GetPrimAtPath(SERVER_PRIM_PATH).IsValid()
        if has_server != args.restaurant_server:
            raise RuntimeError("server presence must match explicit --restaurant-server opt-in")
        fixture_specs = []  # Real dressed meshes remain in their saved additions layer.
    if args.fixture_layout == "three":
        fixture_specs.extend([
            {"name": "raised", "offset_xy": (2.6, 1.5), "drop_m": 0.30,
             "color": (0.85, 0.25, 0.15), "plume": False},
            {"name": "plume", "offset_xy": (3.0 if args.fixture_spacing == "separated" else 2.2, -0.65), "drop_m": 0.0,
             "color": (0.2, 0.8, 0.3), "plume": True},
        ])
    fixtures = []
    actor_translate = None
    actor_start_time = None
    root_before = stage.GetRootLayer().ExportToString()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        puffs = []
        for spec in fixture_specs:
            xy = np.array(stationary.ROBOT_START_POSITION_WORLD[:2]) + spec["offset_xy"]
            base = f"/World/CRALiveObject/{spec['name']}"
            sphere = (UsdGeom.Capsule.Define(stage, base + "/Capsule") if args.actor_shape == "upright_capsule"
                      else UsdGeom.Sphere.Define(stage, base + "/Sphere"))
            center_z = ACTOR_CENTER_Z + spec["drop_m"]
            if args.actor_shape == "upright_capsule":
                sphere.CreateAxisAttr("Z")
                sphere.CreateHeightAttr(ACTOR_HEIGHT - 2 * RADIUS)
                sphere.CreateExtentAttr([Gf.Vec3f(-RADIUS, -RADIUS, -ACTOR_CENTER_Z),
                                         Gf.Vec3f(RADIUS, RADIUS, ACTOR_CENTER_Z)])
            sphere.CreateRadiusAttr(RADIUS)
            translate = sphere.AddTranslateOp()
            translate.Set(Gf.Vec3d(*xy, center_z))
            if spec["name"] == "ground":
                actor_translate = translate
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*spec["color"])])
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            material = UsdShade.Material.Define(stage, base + "/Matte")
            shader = UsdShade.Shader.Define(stage, base + "/Matte/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*spec["color"]))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
            shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
            fixtures.append({**spec, "center_world_xy": xy.tolist(), "radius_m": RADIUS,
                             "center_world_xyz": [*xy.tolist(), center_z], "actor_shape": args.actor_shape,
                             "authored_height_rendering_only_m": ACTOR_HEIGHT,
                             "monitor_radius_m": 0.7 / math.sqrt(2) if spec["drop_m"] else RADIUS})
            if spec["drop_m"]:
                support = UsdGeom.Cube.Define(stage, base + "/Support")
                support.CreateSizeAttr(1.0)
                support.AddTranslateOp().Set(Gf.Vec3d(*xy, spec["drop_m"] / 2))
                support.AddScaleOp().Set(Gf.Vec3f(0.7, 0.7, spec["drop_m"]))
                support.CreateDisplayColorAttr([Gf.Vec3f(0.35)])
                UsdPhysics.CollisionAPI.Apply(support.GetPrim())
            if not spec["plume"]:
                continue
            for index in range(4):
                puff = UsdGeom.Sphere.Define(stage, base + f"/Plume/Puff{index}")
                puff.CreateRadiusAttr((0.035 + 0.009 * index) * (1.7 if args.fixture_layout == "three" else 1.0))
                puff.AddTranslateOp().Set(
                    Gf.Vec3d(xy[0], xy[1] + 0.012 * (-1) ** index,
                             spec["drop_m"] + 2 * RADIUS + 0.07 + 0.085 * index)
                )
                puff.AddScaleOp().Set(Gf.Vec3f(1, 1, 1.4))
                puff.CreateDisplayColorAttr([Gf.Vec3f(0.8)])
                puff.MakeInvisible()
                puffs.append(puff)
    restaurant_props = []
    layout_monitor = None
    if args.restaurant_scene:
        for spec in service_props():
            path = f"{SERVICE_ROOT}/{spec.name}"
            bounds = read_world_bounds(stage, path)
            restaurant_props.append({"name":spec.name,"prim_path":path,
                "bounds_minimum":bounds.minimum,"bounds_maximum":bounds.maximum})
            if spec.name == restaurant_target:
                fixtures.append({"name":spec.name,"center_world_xy":bounds.center[:2],
                    "center_world_xyz":bounds.center,"monitor_radius_m":math.hypot(*bounds.size[:2])/2,
                    "actor_shape":"audited_CC0_restaurant_mesh","plume":False,"drop_m":None})
                if args.restaurant_plume_diagnostic:
                    with Usd.EditContext(stage, stage.GetSessionLayer()):
                        for index in range(4):
                            puff = UsdGeom.Sphere.Define(stage, f"/World/CRARestaurantVisiblePlume/Puff{index}")
                            puff.CreateRadiusAttr(.045 + .01*index)
                            puff.AddTranslateOp().Set(Gf.Vec3d(bounds.center[0], bounds.center[1]+.012*(-1)**index,
                                bounds.maximum[2]+.18+.09*index))
                            puff.AddScaleOp().Set(Gf.Vec3f(1,1,1.4))
                            puff.CreateDisplayColorAttr([Gf.Vec3f(.9)])
                            puff.MakeInvisible()
                            puffs.append(puff)
        if has_server and human is None:
            bounds = read_world_bounds(stage,SERVER_PRIM_PATH)
            restaurant_props.append({"name":"Server","prim_path":SERVER_PRIM_PATH,
                "bounds_minimum":bounds.minimum,"bounds_maximum":bounds.maximum})
            fixtures.append({"name":"Server","center_world_xy":bounds.center[:2],
                "monitor_radius_m":math.hypot(*bounds.size[:2])/2,"actor_shape":"static_staff_avatar",
                "plume":False,"drop_m":None,"matching_scope":"independent planar match; full authored height is not observed geometry"})
        layout_monitor = load_authored_layout_monitor(args.restaurant_layout_survey, restaurant_props)
        save_json("restaurant-placement-monitor.json", {"props":restaurant_props,
            "basis":"authored actual-mesh bounds for independent evaluation/abort only; no factor inputs",
            "humans_added":int(has_server),"actor_motion_enabled":args.restaurant_human in {"crossing", "continuous-crossing"},
            "payload_added":False,"human_carried_bowl_added":args.human_carry_bowl})
        setup_footprint = RectangularRobotFootprint(stationary.ROBOT_FRONT_EXTENT,
            stationary.ROBOT_REAR_EXTENT,stationary.ROBOT_LEFT_EXTENT,
            stationary.ROBOT_RIGHT_EXTENT,stationary.ROBOT_SAFETY_MARGIN)
        setup_pose = (*stationary.ROBOT_START_POSITION_WORLD[:2],stationary.ROBOT_START_YAW_RADIANS)
        setup_clearance = authored_raster_clearance(layout_monitor[0],setup_pose,setup_footprint,layout_monitor[1])
        save_json("restaurant-setup-validation.json",{"passed":setup_clearance>.02,
            "configured_pose_world_xy_yaw":setup_pose,"independent_clearance_m":setup_clearance,
            "scope":"independent authored setup/abort guard; never perception, nominal or safety-filter input"})
        if setup_clearance <= .02:
            raise RuntimeError("restaurant robot setup overlaps/approaches authored occupancy; no perception or wheel trial accepted")
    save_json("fixture-monitor.json", {"fixtures": fixtures, "scope": "authored geometry for rendering and independent validation only; never estimator inputs"})
    root_unchanged_by_fixture = stage.GetRootLayer().ExportToString() == root_before
    controller = NovaCarterController(robot_prim_path=stationary.ROBOT_PRIM_PATH)
    SimulationManager.setup_simulation(
        dt=stationary.PHYSICS_DT, device=stationary.PHYSICS_DEVICE
    )
    timeline = omni.timeline.get_timeline_interface()
    if args.human_carry_bowl:
        carb.settings.get_settings().set("/app/player/useFixedTimeStepping", True)
        timeline.set_play_every_frame(True)
        timeline.set_time_codes_per_second(60)
    timeline.play()
    app_utils.update_app(steps=10)
    if human is not None and not args.human_carry_bowl:
        human.initialize()
        human.start_video()
    controller.verify_articulation()
    if args.human_carry_bowl:
        # Match the validated native component's startup chronology. Pickup
        # before native idle settling can select a different articulated pose.
        yaw = stationary.ROBOT_START_YAW_RADIANS
        controller.robot.set_world_poses(positions=[stationary.ROBOT_START_POSITION_WORLD],
            orientations=[[math.cos(yaw/2), 0., 0., math.sin(yaw/2)]])
    else:
        stationary.position_robot_for_experiment(controller)
    for _ in range(stationary.SETTLE_STEPS):
        controller.stop()
        app.update()
    if args.human_carry_bowl:
        human.start_video()
        for _ in range(30):
            controller.stop()
            app.update()
        human.initialize()
        for _ in range(60):
            controller.stop()
            app.update()
        human.begin_carry()
        for _ in range(180):
            controller.stop()
            app.update()
            pose = stationary.read_robot_pose(stage)
            human.update(float(timeline.get_current_time()), (pose.x, pose.y), (0., 0.), layout_monitor)
            if human.carry.ready:
                break
        if not human.carry.ready:
            raise RuntimeError("native stationary hand attachment failed before perception")
        save_json("native-carry-startup-audit.json", {
            "robot_settle_steps": stationary.SETTLE_STEPS, "camera_settle_steps": 30,
            "native_idle_settle_steps": 60, "pickup_ready_time": float(timeline.get_current_time()),
            "pickup_status": str(human.agent.get_task_status(human.carry.task)),
            "scope": "initial zero-wheel native component chronology; no scheduled observation pauses during navigation"})
    anchor = stationary.read_robot_pose(stage)
    if layout_monitor is not None:
        settled_clearance = authored_raster_clearance(layout_monitor[0],(anchor.x,anchor.y,anchor.yaw),setup_footprint,layout_monitor[1])
        if settled_clearance <= .02:
            controller.stop()
            raise RuntimeError("settled restaurant setup fails independent clearance guard")
    active_waypoint_local = ROUTE[0]
    save_json("fixture-monitor.json", {"fixtures": fixtures,
        "anchor_world_xy_yaw": [anchor.x, anchor.y, anchor.yaw],
        "scope": "authored fixture and oracle anchor for independent evaluation/rendering only; never factor input"})
    from robot.scene.camera_mount import apply_experimental_camera_pitch
    save_json("camera-mount-audit.json", apply_experimental_camera_pitch(
        stage, NovaCarterCamera().camera_prim_path,
        args.human_camera_upward_degrees, stationary.ROBOT_BASE_PRIM_PATH,
        raise_metres=args.human_camera_raise_metres,
    ))
    calibration = read_fisheye_polynomial_calibration(
        stage,
        camera_prim_path=NovaCarterCamera().camera_prim_path,
        output_resolution=(400, 640) if args.restaurant_scene else (480, 640),
    )
    if args.restaurant_scene and (
        calibration.output_width * calibration.nominal_height
        != calibration.output_height * calibration.nominal_width
    ):
        raise RuntimeError("restaurant RGB acquisition must match the authored calibration aspect ratio")
    save_json("camera-calibration.json", asdict(calibration))
    camera, lidar = (
        NovaCarterCamera(resolution=(400,640) if args.restaurant_scene else (480,640)),
        NovaCarterLidar(lidar_prim_path=stationary.PANDAR_XT32_LIDAR_PRIM_PATH),
    )
    camera.initialize()
    lidar.initialize()
    robot_from_lidar = compute_relative_prim_transform(
        stage,
        source_prim_path=stationary.PANDAR_XT32_LIDAR_PRIM_PATH,
        target_prim_path=stationary.ROBOT_BASE_PRIM_PATH,
    )
    camera_from_robot = compute_relative_prim_transform(
        stage,
        source_prim_path=stationary.ROBOT_BASE_PRIM_PATH,
        target_prim_path=camera.camera_prim_path,
    )
    policies = existing.create_fixed_field_configuration()
    class_policy, guidance, grid, field_config = policies
    grid = replace(grid, resolution=0.1)
    field_config = replace(
        field_config,
        configuration_space_config=replace(
            field_config.configuration_space_config, yaw_slice_count=16,
            expansion_backend="vectorized" if args.vectorized_fields else "loop",
        ),
        h_full_config=replace(field_config.h_full_config, yaw_slice_count=16),
        guidance_field_config=replace(
            field_config.guidance_field_config,
            allow_dense_scene_arbitration=True,
            solver_backend="sparse_direct",
            interface_backend="vectorized",
        ),
    )
    field_config = replace(
        field_config,
        h_full_config=replace(
            field_config.h_full_config,
            poisson_config=replace(
                field_config.h_full_config.poisson_config,
                solver_backend="sparse_direct",
            ),
        ),
    )
    from robot.perception.object_height import ObjectHeightConfig
    from robot.perception.human_body_contact import HumanBodyContactPriorConfig
    from robot.safety.motion_aware_passing import MotionAwarePassingConfig
    from robot.perception.human_body_envelope import HumanBodyEnvelopeConfig

    perception = LiveObjectPerception(
        segmenter=segmenter,
        mass_estimator=mass,
        contact_detector=contact,
        person_detector=person,
        human_mass_prior_config=HumanMassPriorConfig(
            "bounded adult-human navigation; visually associated configured mass interval, not calibration",
            mass_kg=args.human_mass_prior[0], lower_kg=args.human_mass_prior[1], upper_kg=args.human_mass_prior[2],
        ) if person is not None else None,
        human_body_contact_config=HumanBodyContactPriorConfig(
            score=args.human_body_contact_prior,
        ) if args.human_body_contact_prior is not None else None,
        experiment_baseline=args.experiment_baseline,
        motion_passing_config=MotionAwarePassingConfig(require_visual_person_selection=args.human_observed_passing_target,
            maximum_identity_prior_selection_age_seconds=args.human_passing_identity_prior_max_age) if args.motion_aware_passing else None,
        human_body_envelope_config=HumanBodyEnvelopeConfig(body_radius_m=args.human_body_envelope_radius,
            centroid_uncertainty_m=args.human_envelope_centroid_uncertainty,
            maximum_identity_age_seconds=args.human_envelope_identity_max_age) if args.human_body_envelope_radius is not None else None,
        calibration=calibration,
        class_policy=class_policy,
        guidance_policy=guidance,
        grid_config=grid,
        field_config=field_config,
        anchor_position_world_xy=(anchor.x, anchor.y),
        anchor_yaw=anchor.yaw,
        ground_prior_scope=GROUND_SCOPE if args.actor_shape == "sphere" and not args.restaurant_scene and not args.object_height_evidence else None,
        negative_plume_scope=("bounded stationary native-aspect real-bowl view: explicitly clean reference phase; object-associated visible cue only, not absence of heat"
            if args.restaurant_plume_diagnostic else None) if args.restaurant_scene else PLUME_SCOPE,
        rectify_plume=True,
        occupancy_retention_seconds=0.15 if args.kinetic_diagnostic or args.moving_actor or human is not None else 0.5,
        static_occupancy_memory_prior_scope=("bounded static dressed restaurant and stationary staff avatar; positive observed occupancy retained; no moving people/assets or free-space clearing"
            if args.restaurant_scene else "bounded static sphere/environment; positive occupancy remains until experiment end; no dynamic-actor or free-space-clearing claim")
        if args.static_occupancy_memory_prior
        else None,
        evidence_retention_config=None
        if args.disable_evidence_retention or args.persistent_object_risk_map
        else TemporalObjectEvidenceConfig(),
        persistent_property_config=PersistentObjectEvidenceConfig(
            "bounded one-adult navigation: only intrinsic mass may persist for 3 s; articulated shape/contact, support, plume and motion remain current",
            maximum_property_age_seconds=3.0,
            confidence_half_life_seconds=1.5,
            retained_properties=("mass",),
            translate_instance_bounds_prior=False,
            allow_instance_bounds_prior=False,
            minimum_visual_reassessment_interval_seconds=args.visual_reassessment_interval,
            maximum_visual_tracks_per_update=args.live_visual_tracks,
        ) if human is not None else PersistentObjectEvidenceConfig(
            "bounded static restaurant: acquired intrinsic properties persist with original expiry; potential/thermal unsupported inputs retain configured priors"
            if args.restaurant_scene else "bounded separated fixture: mass/visible contact/plume assumed unchanged; support invalidates on displacement; current motion/route; moving actor uses short occupancy expiry",
            translate_instance_bounds_prior=args.kinetic_diagnostic or args.moving_actor,
            minimum_visual_reassessment_interval_seconds=args.visual_reassessment_interval,
            maximum_visual_tracks_per_update=args.live_visual_tracks,
        )
        if args.persistent_object_risk_map
        else None,
        object_factor_prior_config=ObjectRiskPriorConfig(
            PhysicalRiskVector(
                args.object_factor_prior_risk,
                args.object_factor_prior_risk,
                args.object_factor_prior_risk,
                args.object_factor_prior_risk,
            )
        )
        if args.persistent_object_risk_map
        else None,
        fall_surface_config=LidarFallSurfaceConfig(
            support_xy_margin_m=0.30, support_maximum_gap_m=0.20,
            minimum_support_overhang_m=0.05, landing_maximum_offset_m=2.5,
            allow_equal_height_multiple_landings=True,
        ) if args.fixture_layout == "three" and not args.object_height_evidence else None,
        fall_interaction_prior_scope="bounded multi-object support test; worst-case harmful fall interaction, not measured fall probability"
        if args.fixture_layout == "three" or args.object_height_evidence else None,
        object_height_config=ObjectHeightConfig(surface_config=LidarFallSurfaceConfig(
            support_xy_margin_m=args.height_support_margin,
            support_maximum_gap_m=args.height_support_gap,
        )) if args.object_height_evidence else None,
        cluster_in_3d=args.fixture_layout == "three" or args.actor_shape == "upright_capsule" or bool(args.restaurant_scene),
        cluster_radius_m=args.human_cluster_radius if human is not None else None,
        camera_visibility_radius_pixels=4 if args.fixture_layout == "three" or args.actor_shape == "upright_capsule" or args.restaurant_scene else 0,
        mask_foreground_depth_gap_m=0.5 if args.fixture_layout == "three" or args.actor_shape == "upright_capsule" or args.restaurant_scene else None,
        plume_view_fov_degrees=args.plume_view_fov,
    )
    history = EgoPoseHistory()
    captures, rejections, records = [], [], []
    last_sensor_id = -1
    copied_scan = None
    navigation_active = False

    def state():
        pose = stationary.read_robot_pose(stage)
        dx, dy = pose.x - anchor.x, pose.y - anchor.y
        c, s = math.cos(anchor.yaw), math.sin(anchor.yaw)
        return np.array(
            [
                c * dx + s * dy,
                -s * dx + c * dy,
                (pose.yaw - anchor.yaw + math.pi) % math.tau - math.pi,
            ]
        )

    def step(command=(0.0, 0.0), paced=False):
        started = time.perf_counter()
        if actor_start_time is not None:
            offset, _, _ = actor_offset_velocity(max(0, float(timeline.get_current_time()) - actor_start_time))
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                actor_translate.Set(Gf.Vec3d(center_world[0], center_world[1] + offset, ACTOR_CENTER_Z))
        controller.apply_command(*command)
        app.update()
        now = float(timeline.get_current_time())
        matrix = compute_relative_prim_transform(
            stage,
            source_prim_path=stationary.ROBOT_BASE_PRIM_PATH,
            target_prim_path="/World",
        )
        history.append(
            EgoPoseSample(now, matrix, EstimateStatus.ORACLE, "isaac_usd_chassis_pose")
        )
        if human is not None:
            human.update(now, (matrix[0, 3], matrix[1, 3]), command, layout_monitor,
                         navigation=navigation_active)
            if human.carry is not None and human.carry.ready:
                pose = stationary.read_robot_pose(stage)
                bowl = human.carry.records[-1]
                clearance = min(
                    disk_footprint_clearance((pose.x, pose.y, pose.yaw),
                        center_xy=human.position()[:2], radius_m=human.setup["collision_radius_m"],
                        footprint=setup_footprint),
                    disk_footprint_clearance((pose.x, pose.y, pose.yaw),
                        center_xy=bowl["bowl_center_xyz"][:2],
                        radius_m=float(np.linalg.norm(bowl["world_half_extents"][:2])),
                        footprint=setup_footprint))
                human.records[-1]["independent_robot_clearance_m"] = clearance
                if clearance <= .02:
                    controller.stop()
                    raise RuntimeError("independent body/bowl robot-clearance guard")
        if paced:
            time.sleep(
                max(
                    0.0,
                    stationary.PHYSICS_DT / args.simulation_rate
                    - (time.perf_counter() - started),
                )
            )
        return now

    def capture(clean=False):
        nonlocal last_sensor_id, copied_scan
        if copied_scan is None:
            raw = lidar.get_point_cloud().copy()
            info = lidar.get_frame_info()
            if info.sensor_frame_id <= last_sensor_id:
                return None
            rgb = camera.get_rgb().copy()
            camera_info = camera.get_frame_info()
            metadata = lidar.get_scan_metadata()
            if human is not None:
                copied_scan = (raw, info, rgb, camera_info, metadata, float(timeline.get_current_time()))
                last_sensor_id = info.sensor_frame_id
        if copied_scan is not None:
            raw, info, rgb, camera_info, metadata, queued_time = copied_scan
            offsets = metadata.point_time_offsets_nanoseconds
            now = float(timeline.get_current_time())
            if offsets is None or not len(offsets) or now - queued_time > .05 + 1e-9:
                copied_scan = None
                raise ValueError("copied human scan missing timing or exceeded .05 s pose wait")
            if info.sensor_timestamp_nanoseconds * 1e-9 + int(offsets.max()) * 1e-9 > history.samples[-1].timestamp_seconds + 1e-9:
                return None
            copied_scan = None
        scan_audit = metadata.summary()
        for name, scan_pose in (
            ("start", metadata.frame_start),
            ("end", metadata.frame_end),
        ):
            if scan_pose is None:
                continue
            try:
                from scipy.spatial.transform import Rotation

                sensor_pose = (
                    history.sample(
                        scan_pose.timestamp_nanoseconds * 1e-9
                    ).world_from_robot
                    @ robot_from_lidar
                )
                rotation = Rotation.from_quat(scan_pose.orientation_xyzw).as_matrix()
                angle = Rotation.from_matrix(
                    sensor_pose[:3, :3].T @ rotation
                ).magnitude()
                scan_audit[f"{name}_usd_position_error_m"] = float(
                    np.linalg.norm(sensor_pose[:3, 3] - scan_pose.position_m)
                )
                scan_audit[f"{name}_usd_rotation_error_radians"] = float(angle)
            except ValueError as error:
                scan_audit[f"{name}_pose_comparison_unavailable"] = str(error)
        scan_time = info.sensor_timestamp_nanoseconds * 1e-9
        epoch = align_camera_lidar_epoch(
            history,
            lidar_timestamp_seconds=scan_time,
            camera_timestamp_seconds=camera_info.simulation_time_seconds,
            camera_from_robot=camera_from_robot,
        )
        velocity = history.velocity_world_xy(scan_time)
        if args.disable_lidar_deskew:
            scene = transform_points(filter_valid_points(raw), robot_from_lidar)
            scan_audit["ego_motion_correction"] = {
                "method": "legacy_scan_start_transform"
            }
        else:
            scene, scan_audit["ego_motion_correction"] = deskew_uncompensated_lidar(
                raw,
                metadata,
                history,
                robot_from_sensor=robot_from_lidar,
                reference_timestamp_seconds=scan_time,
            )
        pose = epoch.lidar_pose.world_from_robot
        yaw = math.atan2(pose[1, 0], pose[0, 0])
        gravity_from_body = gravity_aligned_robot_transform(pose)
        body_from_gravity = np.linalg.inv(gravity_from_body)
        scene = transform_points(scene, gravity_from_body)
        floor = -float(pose[2, 3])
        points = detect_obstacle_points_3d(
            scene,
            ObstacleDetectionConfig(
                minimum_height=floor + 0.05,
                maximum_height=floor + (1.8 if human is not None else 1.0),
                minimum_range=0.15,
                maximum_range=7.5 if human is not None else 5.0,
                forward_only=False,
                angular_bin_size_degrees=None,
            ),
        )
        c, s = math.cos(anchor.yaw), math.sin(anchor.yaw)
        goal_world = (
            anchor.x + c * active_waypoint_local[0] - s * active_waypoint_local[1],
            anchor.y + s * active_waypoint_local[0] + c * active_waypoint_local[1],
        )
        packet = LiveObjectPacket(
            info.sensor_frame_id,
            scan_time,
            camera_info.read_index,
            camera_info.simulation_time_seconds,
            rgb,
            scene,
            points,
            tuple(pose[:2, 3]),
            yaw,
            velocity,
            epoch.camera_from_scan_robot @ body_from_gravity,
            floor,
            goal_world,
            clean,
            epoch.lidar_pose.source,
            epoch.lidar_pose.status,
            (
                np.stack(
                    [pose]
                    + [
                        item.world_from_robot
                        for item in history.samples
                        if scan_time
                        < item.timestamp_seconds
                        <= camera_info.simulation_time_seconds
                    ]
                )
                if args.disable_lidar_deskew
                else np.stack([pose])
            )
            @ body_from_gravity,
            "bounded static scene; legacy conservative scan-interval union, not per-return deskew"
            if args.disable_lidar_deskew
            else "bounded static scene; per-return ego-motion deskew to retained scan-header pose; no independently moving object correction",
        )
        last_sensor_id = info.sensor_frame_id
        captures.append(
            {
                "lidar_id": info.sensor_frame_id,
                "lidar_time": scan_time,
                "camera_id": camera_info.read_index,
                "camera_time": camera_info.simulation_time_seconds,
                "state": state().tolist(),
                "scan_metadata": scan_audit,
            }
        )
        if args.save_sensor_snapshots and (
            len(captures) <= 24 or len(captures) % 20 == 0
        ):
            snapshot = args.output / "sensor-snapshots"
            snapshot.mkdir(exist_ok=True)
            stem = f"{info.sensor_frame_id:06d}"
            cv2.imwrite(str(snapshot / f"{stem}.png"), rgb[:, :, ::-1])
            np.savez_compressed(
                snapshot / f"{stem}.npz",
                raw_sensor_xyz=raw,
                scene_xyz=scene,
                obstacle_xyz=points,
                point_time_offsets_ns=np.empty(0, dtype=np.int64)
                if metadata.point_time_offsets_nanoseconds is None
                else metadata.point_time_offsets_nanoseconds,
                point_timing_available=metadata.point_time_offsets_nanoseconds
                is not None,
                robot_from_lidar=robot_from_lidar,
                world_from_robot=pose,
                camera_from_scan_robot=packet.camera_from_scan_robot,
            )
            (snapshot / f"{stem}.json").write_text(
                json.dumps(captures[-1], indent=2, default=json_default) + "\n",
                encoding="utf-8",
            )
        return packet

    # Stationary calibration/development gate, distinct from navigation.
    # Let the RGB renderer/reference domain settle before its clean reference.
    for _ in range(60):
        step()
    latest = None
    observation_frames = args.stationary_frames * (3 if args.restaurant_plume_diagnostic else 1)
    for _ in range(max(1000 if args.restaurant_plume_diagnostic else 400,
                       observation_frames * 10 + 40)):
        now = step()
        try:
            packet = capture(clean=(len(records) < args.stationary_frames
                if args.restaurant_plume_diagnostic else not records))
        except (RuntimeError, ValueError) as error:
            rejections.append(str(error))
            continue
        if packet is None:
            continue
        latest = perception(packet)
        print(
            json.dumps(
                {
                    "stationary_frame": len(records) + 1,
                    "build_wall_seconds": latest.field.build_wall_seconds,
                    "readiness": latest.readiness.ready,
                    "unknown": latest.diagnostics.get("unknown_factors"),
                    "timings": latest.diagnostics.get("stage_wall_seconds"),
                    "objects": len(latest.objects),
                    "thermal_evidence": latest.attributes.thermal.heat_source_likelihood.evidence,
                },
                default=json_default,
            ),
            flush=True,
        )
        records.append(field_record(latest, now))
        if args.restaurant_plume_diagnostic:
            phase = (len(records)-1)//args.stationary_frames
            records[-1]["restaurant_plume_phase"] = ("clean_reference_validation", "no_plume", "visible_plume")[phase]
            records[-1]["stationary_rgb_pose"] = state().tolist()
            records[-1]["camera_frame_id"] = packet.camera_frame_id
            directory = args.output/"plume-rgb"
            directory.mkdir(exist_ok=True)
            cv2.imwrite(str(directory/f"{len(records):03d}.png"), packet.rgb[:,:,::-1])
            if len(records) == 2*args.stationary_frames:
                cv2.imwrite(str(args.output/"no-plume-current.png"), packet.rgb[:,:,::-1])
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    for puff in puffs:
                        puff.MakeVisible()
        save_instance_snapshot(latest)
        if len(records) == 1:
            cv2.imwrite(
                str(args.output / "clean-reference.png"), packet.rgb[:, :, ::-1]
            )
            if not args.restaurant_scene and not (
                latest.diagnostics.get("accepted_instance")
                or latest.diagnostics.get("accepted_instance_count", 0)
            ):
                raise RuntimeError(
                    "stationary clean reference lacks an accepted instance"
                )
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                if args.plume_control == "persistent":
                    for puff in puffs:
                        puff.MakeVisible()
        if len(records) >= observation_frames:
            cv2.imwrite(
                str(args.output / "stationary-current.png"), packet.rgb[:, :, ::-1]
            )
            break
    save_json("stationary-fields.json", records)
    save_json("captures.json", captures)
    stationary_ready = (
        latest is not None and len(records) >= 12 and latest.readiness.ready
    )
    if args.persistent_object_risk_map:
        stationary_ready = (
            latest is not None
            and len(records) >= 12
            and all(
                record["diagnostics"]["controller_numeric_policy_ready"]
                for record in records
            )
            and any(
                record["diagnostics"]["accepted_instance_count"] > 0
                for record in records
            )
        )
    gate = {
        "passed": stationary_ready,
        "frames": len(records),
        "nonzero_wheel_commands": 0,
        "plume_control": args.plume_control,
        "fixture_layout": args.fixture_layout,
        "fixture_spacing": args.fixture_spacing,
        "actor_shape": "real_restaurant_meshes" if args.restaurant_scene else args.actor_shape,
        "plume_view_fov_degrees": args.plume_view_fov,
        "rejections": sorted(set(rejections)),
        "weights": weights,
        "last_readiness": None if latest is None else asdict(latest.readiness),
        "persistent_object_risk_map": args.persistent_object_risk_map,
        "last_objects": []
        if latest is None
        else [
            {
                "track_id": item.attributes.track_id,
                "phi": item.policy.risk,
                "perception_ready": item.readiness.ready,
                "factor_scores": item.diagnostics["factor_scores"],
            }
            for item in latest.objects
        ],
        "scope": "stationary all-track numeric-policy dataflow with accepted visual observations and configured factor priors; not complete visual estimation for every track"
        if args.persistent_object_risk_map
        else "live fresh stationary RGB/LiDAR; category-free rough mass, explicit local ground/fall interaction priors, visual contact/plume proxies",
    }
    if args.restaurant_scene:
        object_summary = summarize_fixture_assessments(records, fixtures, maximum_distance_m=0.20)
        if args.restaurant_server and human is None:
            server_summary = summarize_fixture_assessments(records,[f for f in fixtures if f["name"]=="Server"],maximum_distance_m=.40)
            object_summary["fixtures"]["Server"] = server_summary["fixtures"]["Server"]
            object_summary["server_matching_scope"] = "independent planar .40 m partial-body match; unchanged mask/ownership gates; not calibrated whole-human geometry"
        gate["restaurant_component_checks"] = restaurant_asset_checks(object_summary, restaurant_target)
        gate["restaurant_observation"] = args.restaurant_view
        gate["fixture_gate_scope"] = f"real {restaurant_target} mesh: owned RGB/LiDAR mass, contact proxy and motion; potential/thermal may use labeled priors; not full four-factor sensor acceptance"
        if human is None:
            gate["passed"] = stationary_ready = bool(stationary_ready and all(gate["restaurant_component_checks"].values()))
        else:
            gate["other_restaurant_components_scope"] = "bowl diagnostics retained; the selected stationary component is the human; unsupported other tracks retain occupancy and labeled factor priors"
        if args.restaurant_server and human is None:
            gate["server_component_checks"] = restaurant_asset_checks(object_summary,"Server")
            gate["passed"] = stationary_ready = bool(stationary_ready and all(gate["server_component_checks"].values()))
        save_json("stationary-object-validation.json", object_summary)
        if human is not None:
            from robot.evaluation.human_navigation_validation import human_stationary_checks
            gate["human_component_checks"] = human_stationary_checks(records)
            if args.human_body_envelope_radius is not None:
                from robot.evaluation.human_body_envelope_validation import validate_human_body_envelopes
                envelope_validation = validate_human_body_envelopes(records, human.records,
                    reference_radius_m=human.setup["collision_radius_m"])
                save_json("stationary-human-envelope-validation.json", envelope_validation)
                gate["human_body_envelope_checks"] = envelope_validation["checks"]
                stationary_ready = stationary_ready and envelope_validation["passed"]
            gate["passed"] = stationary_ready = bool(stationary_ready
                and all(gate["human_component_checks"].values()))
            gate["fixture_gate_scope"] = "fresh visually associated adult mass prior/current motion/contact proxy; potential and thermal priors; wheels still blocked until this gate passes"
        if args.restaurant_plume_diagnostic:
            plume_validation = restaurant_plume_checks(records, fixtures)
            save_json("restaurant-plume-validation.json", plume_validation)
            gate["restaurant_plume_checks"] = plume_validation["checks"]
            gate["passed"] = stationary_ready = bool(stationary_ready and plume_validation["passed"])
    elif args.fixture_layout == "three" or args.moving_actor or args.actor_shape == "upright_capsule":
        object_summary = summarize_fixture_assessments(records, fixtures, maximum_distance_m=0.25)
        gate["bounded_fixture_checks"] = (kinetic_actor_fixture_checks(object_summary)
            if args.actor_shape == "upright_capsule" else bounded_fixture_checks(object_summary))
        gate["fixture_gate_scope"] = ("kinetic-component gate with explicit potential prior; not complete four-factor perception or calibrated human estimation"
            if args.actor_shape == "upright_capsule" else "sphere/support/plume fixture evidence gate")
        gate["passed"] = stationary_ready = bool(stationary_ready and gate["bounded_fixture_checks"] and all(gate["bounded_fixture_checks"].values()))
        save_json("stationary-object-validation.json", object_summary)
    save_json("stationary-gate.json", gate)
    print(json.dumps({key: value for key, value in gate.items() if key != "last_objects"}, default=json_default), flush=True)
    if not stationary_ready:
        controller.stop()
        raise RuntimeError("stationary restaurant component gate failed; wheels are blocked"
                           if args.restaurant_scene else "stationary four-factor gate failed; wheels are blocked")
    if args.persistent_object_risk_map:
        perception.persistent_properties.freeze_properties()
    if args.kinetic_diagnostic:
        actor_start_time = float(timeline.get_current_time())
        kinetic_records = []
        while float(timeline.get_current_time()) - actor_start_time < 12.1:
            now = step()
            try:
                packet = capture()
                if packet is None:
                    continue
                latest = perception(packet)
            except (ValueError, RuntimeError) as error:
                rejections.append(str(error))
                continue
            elapsed = packet.sensor_timestamp_seconds - actor_start_time
            if elapsed < 0:
                continue
            offset, speed, phase = actor_offset_velocity(elapsed)
            record = field_record(latest, now, state())
            record["actor_monitor"] = {
                "elapsed_seconds": elapsed, "phase": phase,
                "center_world_xy": [center_world[0], center_world[1] + offset],
                "velocity_world_xy": [0, speed],
                "scope": "authored kinematic motion truth for evaluation only; never perception or factor input",
            }
            kinetic_records.append(record)
            if len(kinetic_records) % 20 == 0:
                print(json.dumps({"kinetic_frames": len(kinetic_records), "elapsed": elapsed, "phase": phase}), flush=True)
        report = summarize_kinetic_records(kinetic_records)
        report.update(nonzero_wheel_commands=0, rejections=sorted(set(rejections)),
                      occupancy_retention_seconds=perception.occupancy_retention_seconds,
                      scene_unchanged=original_hash == hashlib.sha256(Path(stationary.ROBOT_SCENE_PATH).read_bytes()).hexdigest())
        report["maximum_robot_translation_m"] = max(np.linalg.norm(record["state_at_commit"][:2]) for record in kinetic_records)
        report["maximum_robot_yaw_radians"] = max(abs(record["state_at_commit"][2]) for record in kinetic_records)
        report["checks"]["stationary_robot"] = bool(report["maximum_robot_translation_m"] < 0.02 and report["maximum_robot_yaw_radians"] < 0.03)
        report["checks"]["scene_invariants"] = report["scene_unchanged"] and root_unchanged_by_fixture
        report["maximum_retained_occupancy_age_seconds"] = max(
            (record["sensor_timestamp"] - item["timestamp"]
             for record in kinetic_records
             for item in record["diagnostics"]["retained_occupancy_epochs"]), default=0.0)
        report["checks"]["bounded_old_occupancy"] = report["maximum_retained_occupancy_age_seconds"] <= 0.15 + 1e-8
        report["passed"] = all(report["checks"].values())
        save_json("kinetic-fields.json", kinetic_records)
        save_json("kinetic-gate.json", report)
        cv2.imwrite(str(args.output / "kinetic-current.png"), packet.rgb[:, :, ::-1])
        print(json.dumps(report), flush=True)
        controller.stop()
        timeline.stop()
        if not report["passed"]:
            raise RuntimeError("moving-object kinetic gate failed")
        return
    if args.evidence_gap_diagnostic:
        # Real fresh RGB/LiDAR continue; explicitly suppress only the visual
        # instance output. This is a fault-injection continuity diagnostic,
        # not a claim that the camera naturally occluded the object.
        original_set_image, original_propose = segmenter.set_image, segmenter.propose
        gap_records, recovery_records = [], []
        segmenter.set_image = lambda rgb: None
        segmenter.propose = lambda *a, **kw: SamInstanceResult(
            None, ("stationary diagnostic injected missing instance output",)
        )
        try:
            for _ in range(200):
                now = step()
                try:
                    packet = capture()
                except (RuntimeError, ValueError):
                    continue
                if packet is None:
                    continue
                latest = perception(packet)
                gap_records.append(field_record(latest, now))
                if len(gap_records) == 1:
                    cv2.imwrite(
                        str(args.output / "gap-current.png"), packet.rgb[:, :, ::-1]
                    )
                if len(gap_records) >= 13:
                    break
        finally:
            segmenter.set_image, segmenter.propose = (
                original_set_image,
                original_propose,
            )
        for _ in range(100):
            now = step()
            try:
                packet = capture()
            except (RuntimeError, ValueError):
                continue
            if packet is None:
                continue
            latest = perception(packet)
            recovery_records.append(field_record(latest, now))
            if len(recovery_records) >= 5:
                break
        retained_mass = sum(
            r["diagnostics"]["temporal_object_evidence"]["mass"]["mode"] == "retained"
            for r in gap_records
        )
        retained_contact = sum(
            r["diagnostics"]["temporal_object_evidence"]["contact"]["mode"]
            == "retained"
            for r in gap_records
        )
        expired_contact = sum(
            "contact" in r["diagnostics"]["unknown_factors"] for r in gap_records
        )
        recovered = latest.diagnostics["accepted_instance"] and latest.readiness.ready
        unchanged = (
            original_hash
            == hashlib.sha256(
                Path(stationary.ROBOT_SCENE_PATH).read_bytes()
            ).hexdigest()
        )
        report = {
            "passed": len(gap_records) == 13
            and retained_mass > 0
            and retained_contact > 0
            and expired_contact > 0
            and recovered
            and unchanged
            and root_unchanged_by_fixture,
            "scope": "stationary fresh sensor data with explicitly injected missing SAM output; no natural occlusion or moving-view acceptance",
            "nonzero_wheel_commands": 0,
            "gap_frames": len(gap_records),
            "retained_mass_frames": retained_mass,
            "retained_contact_frames": retained_contact,
            "contact_unavailable_after_expiry_frames": expired_contact,
            "recovered_four_factor_readiness": bool(recovered),
            "scene_unchanged": unchanged,
            "retention_config": asdict(perception.evidence_memory.config),
        }
        save_json("evidence-gap-fields.json", gap_records)
        save_json("evidence-recovery-fields.json", recovery_records)
        save_json("evidence-gap-report.json", report)
        print(json.dumps(report, default=json_default), flush=True)
        controller.stop()
        timeline.stop()
        if not report["passed"]:
            raise RuntimeError("stationary evidence continuity diagnostic failed")
        return
    if not args.motion:
        controller.stop()
        timeline.stop()
        return

    save_json("numeric-runtime-contract.json", numeric_runtime_contract())

    mpc, backup = existing.create_controller_stack(controller)
    substeps = round(mpc.config.time_step / stationary.PHYSICS_DT)
    mpc = SafeSAGEUnicycleMPC(
        mpc.paper_filter,
        replace(
            mpc.config,
            safety_substeps_per_interval=substeps,
            integration_substeps_per_interval=substeps,
            enable_progress_seeds=True,
            enable_recovery_cost_pruning=True,
            enable_recovery_feasibility_pruning=True,
            maximum_solver_iterations=args.qp_iteration_limit,
            maximum_nonlinear_rescue_iterations=args.nonlinear_rescue_iterations,
            progress_seed_linear_fraction=args.progress_seed_fraction,
            configuration_position_padding_m=0.025,
            configuration_yaw_padding_radians=0.21,
            configuration_padding_initial_fraction=0.1,
        ),
    )
    if args.guidance_tracking_weight > 0:
        mpc = LaplaceGuidanceTrackingMPC(mpc.paper_filter, mpc.config,
            guidance_config=LaplaceGuidanceTrackingConfig(args.guidance_tracking_weight, args.guidance_activation_h,
                math.radians(args.guidance_max_heading_bias_degrees), args.guidance_reference_goal_radius,
                args.guidance_fade_distance, args.guidance_minimum_reference_progress,
                args.guidance_turn_first_minimum_cap, args.guidance_reference_mode))
    control = SafeSAGEControlRuntime(mpc=mpc, backup_filter=backup)
    waypoint = WaypointController(
        [Waypoint(*point) for point in ROUTE],
        config=WaypointControllerConfig(
            maximum_linear_velocity=0.2,
            maximum_angular_velocity=0.6,
            rotate_in_place_threshold_radians=math.radians(100),
            minimum_alignment_scale=0.25,
        ),
    )
    footprint = RectangularRobotFootprint(
        stationary.ROBOT_FRONT_EXTENT,
        stationary.ROBOT_REAR_EXTENT,
        stationary.ROBOT_LEFT_EXTENT,
        stationary.ROBOT_RIGHT_EXTENT,
        stationary.ROBOT_SAFETY_MARGIN,
    )
    buffer = SafeSAGELiveFieldBuffer()

    def commit(result, now):
        snapshot = SafeSAGELiveFieldSnapshot(
            result.field.sensor_frame_id,
            result.field.sensor_timestamp_seconds,
            now,
            result.field.build_wall_seconds,
            result,
        )
        (buffer.commit if buffer.initialized else buffer.initialize)(snapshot)

    commit(latest, float(timeline.get_current_time()))
    coverage_guard = NearbyTrackLossGuard(NearbyTrackLossConfig(
        allow_observed_fragment_merges=human is not None,
        allow_positive_extent_merges=args.human_positive_fragment_extent,
        use_observed_merge_velocity=human is not None)) if args.moving_actor or human is not None else None
    coverage_result = None
    coverage_events = []
    submitted_packet = None

    def coverage_geometry(result, source_packet):
        if human is None:
            return {}
        if source_packet.sensor_frame_id != result.field.sensor_frame_id:
            raise ValueError("coverage guard packet differs from completed source")
        c, s = math.cos(source_packet.robot_yaw), math.sin(source_packet.robot_yaw)
        world = source_packet.obstacle_xyz[:, :2] @ np.array([[c, s], [-s, c]])
        world += source_packet.robot_position_world_xy
        return {"current_points_world_xy": world,
            "current_point_track_ids": result.field.tracking_result.point_track_ids,
            "current_points_source_time": source_packet.sensor_timestamp_seconds}

    if coverage_guard is not None:
        coverage_result = coverage_guard.observe(latest.field.tracking_result.tracked_obstacles,
            source_time=latest.field.sensor_timestamp_seconds, robot_world_xy=(anchor.x, anchor.y),
            **coverage_geometry(latest, packet))
    def infer_with_timing(packet):
        started = time.perf_counter()
        result = profile_call(perception, packet,
            profile_path=args.output / "profiles" / f"perception-{packet.sensor_frame_id}.pstats"
            if args.profile_runtime else None)
        elapsed = time.perf_counter() - started
        if elapsed > 3.0:
            print(json.dumps({"slow_perception_compute": {
                "source_frame_id": packet.sensor_frame_id,
                "source_timestamp": packet.sensor_timestamp_seconds,
                "wall_seconds": elapsed,
                "stage_wall_seconds": result.diagnostics.get("stage_wall_seconds"),
            }}), flush=True)
        return result

    def plan_with_timing(data):
        started = time.perf_counter()
        result = profile_call(control.plan,
            data[0],
            data[1],
            data[2],
            initial_command_guess=data[3],
            **({"nominal_goal_grid_xy": data[4]} if args.guidance_tracking_weight > 0 else {}),
            profile_path=args.output / "profiles" / f"planner-{data[2].sensor_frame_id}.pstats"
            if args.profile_runtime else None,
        )
        if args.save_planner_replay:
            from robot.evaluation.planner_replay import save_planner_replay
            save_planner_replay(args.output / "planner-replays", control, data, result)
        elapsed = time.perf_counter() - started
        if elapsed > 3.0:
            print(json.dumps({"slow_planner_compute": {
                "field_frame_id": data[2].sensor_frame_id,
                "wall_seconds": elapsed,
                "status": result.status,
                "sqp_iteration_count": result.sqp_iteration_count,
                "solver_iteration_count": result.solver_iteration_count,
            }}), flush=True)
        return result

    inference = LiveInferenceRunner(infer_with_timing)
    planner = LiveInferenceRunner(
        plan_with_timing,
        maximum_result_age_seconds=args.planner_result_max_age,
    )
    start_sim = float(timeline.get_current_time())
    navigation_active = True
    if human is not None:
        human.navigation_start_time = start_sim
    if args.moving_actor:
        actor_start_time = start_sim
    start_wall = time.perf_counter()
    plan_id = 0
    plan_waypoints = {}
    waypoint_events = []
    next_plan_due = float(timeline.get_current_time())
    active_plan = None
    active_plan_lease = None
    warm = None
    command = np.zeros(2)
    trajectory, commands, stops, live_fields, plan_events = [], [], [], [], []
    minimum_clearance = math.inf
    minimum_cbf_margin = math.inf
    arrived = False
    reason = "duration_limit"
    consecutive_failures = 0
    def independent_clearance(current):
        clearances = []
        if human is not None:
            xy = human.position()[:2] - [anchor.x, anchor.y]
            c, s = math.cos(anchor.yaw), math.sin(anchor.yaw)
            local = (c * xy[0] + s * xy[1], -s * xy[0] + c * xy[1])
            clearances.append(disk_footprint_clearance(current, center_xy=local,
                radius_m=human.setup["collision_radius_m"], footprint=footprint))
            if human.carry is not None and human.carry.records:
                bowl = human.carry.records[-1]
                xy = np.array(bowl["bowl_center_xyz"][:2]) - [anchor.x, anchor.y]
                local = (c*xy[0]+s*xy[1], -s*xy[0]+c*xy[1])
                clearances.append(disk_footprint_clearance(current, center_xy=local,
                    radius_m=float(np.linalg.norm(bowl["world_half_extents"][:2])), footprint=footprint))
        for fixture in fixtures:
            xy = np.array(fixture["center_world_xy"])
            if args.moving_actor and fixture["name"] == "ground":
                xy[1] += actor_offset_velocity(max(0, float(timeline.get_current_time()) - actor_start_time))[0]
            delta = xy - [anchor.x, anchor.y]
            c, s = math.cos(anchor.yaw), math.sin(anchor.yaw)
            local = np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])
            clearances.append(disk_footprint_clearance(current, center_xy=local,
                radius_m=fixture["monitor_radius_m"], footprint=footprint))
        clearance = min(clearances)
        if layout_monitor is not None:
            pose = stationary.read_robot_pose(stage)
            clearance = min(clearance, authored_raster_clearance(
                layout_monitor[0], (pose.x,pose.y,pose.yaw), footprint, layout_monitor[1]))
        return clearance

    best_goal_distance = math.inf
    last_goal_progress = start_sim
    try:
        while float(timeline.get_current_time()) - start_sim < args.maximum_seconds:
            now = float(timeline.get_current_time())
            current = state()
            goal_distance = float(np.linalg.norm(np.asarray(GOAL) - current[:2]))
            if goal_distance < best_goal_distance - 0.025:
                best_goal_distance, last_goal_progress = goal_distance, now
            elif args.progress_timeout_seconds and now - last_goal_progress > args.progress_timeout_seconds:
                reason = "progress_timeout"
                break
            clearance = independent_clearance(current)
            minimum_clearance = min(minimum_clearance, clearance)
            if clearance <= 0.02:
                reason = "clearance_guard"
                break
            nominal = waypoint.compute_command(RobotPose2D(*current))
            if nominal.route_complete:
                arrived, reason = True, "goal_reached"
                break
            active_waypoint_local = (nominal.target_waypoint.x, nominal.target_waypoint.y)
            if nominal.waypoint_advanced:
                waypoint_events.append({"time": now-start_sim,
                    "next_waypoint_index": nominal.target_waypoint_index, "state": current.tolist()})
                active_plan, active_plan_lease, warm = None, None, None
                next_plan_due = now
            outcome = inference.poll(current_sensor_time_seconds=now)
            if outcome is not None:
                if outcome.value is not None:
                    if coverage_guard is not None:
                        c, s = math.cos(anchor.yaw), math.sin(anchor.yaw)
                        coverage_result = coverage_guard.observe(outcome.value.field.tracking_result.tracked_obstacles,
                            source_time=outcome.value.field.sensor_timestamp_seconds,
                            robot_world_xy=(anchor.x+c*current[0]-s*current[1], anchor.y+s*current[0]+c*current[1]),
                            **coverage_geometry(outcome.value, submitted_packet))
                        if coverage_result["observed_fragment_merges"] or coverage_result["blocked"]:
                            if any(e.get("coexistence_basis") in {"retained_original_coobservation", "recent_original_coobservation_positive_extent"}
                                   for e in coverage_result["observed_fragment_merges"]):
                                audit_name = f"coverage-merge-{outcome.value.field.sensor_frame_id}.npz"
                                np.savez_compressed(args.output / audit_name,
                                    **coverage_geometry(outcome.value, submitted_packet))
                                coverage_result["current_positive_geometry_audit_file"] = audit_name
                            coverage_events.append(coverage_result)
                        if coverage_result["blocked"]:
                            reason = "nearby_track_lost"
                            controller.stop()
                            save_json("unpublished-coverage-stop-field.json", field_record(outcome.value, now, current))
                            if human is not None:
                                np.savez_compressed(args.output / "unpublished-coverage-source.npz",
                                    **coverage_geometry(outcome.value, submitted_packet))
                            print(json.dumps({"coverage_stop": coverage_result, "sim_seconds": now-start_sim}), flush=True)
                            break
                    consecutive_failures = 0
                    commit(outcome.value, now)
                    latest = outcome.value
                    live_fields.append(field_record(latest, now, current))
                    if args.moving_actor:
                        elapsed = latest.field.sensor_timestamp_seconds - actor_start_time
                        offset, speed, phase = actor_offset_velocity(max(0, elapsed))
                        live_fields[-1]["actor_monitor"] = {"elapsed_seconds": elapsed, "phase": phase,
                            "center_world_xy": [center_world[0], center_world[1] + offset],
                            "velocity_world_xy": [0, speed], "scope": "independent authored monitor only"}
                    if args.stream_live_records:
                        with (args.output / "live-fields.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(live_fields[-1], default=json_default, allow_nan=False, separators=(",", ":")) + "\n")
                    if len(live_fields) <= 12 or len(live_fields) % 20 == 0:
                        save_instance_snapshot(latest)
                    if len(live_fields) == 1 or len(live_fields) % 20 == 0:
                        save_json("live-fields.json", live_fields[-5:] if args.stream_live_records else live_fields)
                        save_json("commands.json", commands)
                        if packet is not None and packet.camera_frame_id == latest.detections.source_frame_id:
                            cv2.imwrite(
                                str(args.output / "live-current.png"),
                                packet.rgb[:, :, ::-1],
                            )
                    if len(live_fields) % 20 == 0:
                        print(
                            json.dumps(
                                {
                                    "live_fields": len(live_fields),
                                    "sim_seconds": now - start_sim,
                                    "state": current.tolist(),
                                    "phi": latest.policy.risk,
                                    "phi_scope": "first geometric track only",
                                    "all_object_phi_range": [min(item.policy.risk for item in latest.objects), max(item.policy.risk for item in latest.objects)] if latest.objects else None,
                                    "unknown": latest.diagnostics.get(
                                        "unknown_factors"
                                    ),
                                    "objects": len(latest.objects),
                                    "stage_wall_seconds": latest.diagnostics.get("stage_wall_seconds"),
                                }
                            ),
                            flush=True,
                        )
                else:
                    rejections.append(outcome.rejection_reason)
                    consecutive_failures += 1
                    print(
                        json.dumps(
                            {
                                "inference_rejected": outcome.rejection_reason,
                                "sim_seconds": now - start_sim,
                                "source_age_seconds": now - outcome.sensor_timestamp_seconds,
                                "job_to_poll_wall_seconds": outcome.wall_seconds,
                            }
                        ),
                        flush=True,
                    )
                    if consecutive_failures >= 5:
                        reason = "repeated_inference_failure"
                        break
            if not inference.busy:
                try:
                    packet = capture()
                    if packet is not None:
                        inference.submit(
                            frame_id=packet.sensor_frame_id,
                            sensor_timestamp_seconds=packet.sensor_timestamp_seconds,
                            packet=packet,
                        )
                        submitted_packet = packet
                except (RuntimeError, ValueError) as error:
                    rejections.append(str(error))
            plan_outcome = planner.poll(current_sensor_time_seconds=now)
            if plan_outcome is not None:
                result = plan_outcome.value
                accepted = (
                    result is not None
                    and plan_waypoints.get(plan_outcome.frame_id) == waypoint.current_waypoint_index
                    and result.success
                    and not result.fallback_active
                    and result.minimum_dtcbf_margin >= -mpc.config.constraint_tolerance
                    and result.minimum_predicted_safety_value
                    >= -mpc.config.constraint_tolerance
                )
                plan_events.append(
                    {
                        "time": now - start_sim,
                        "accepted": bool(accepted),
                        "request_waypoint_index": plan_waypoints.pop(plan_outcome.frame_id, None),
                        "wall_seconds": plan_outcome.wall_seconds,
                        "reason": plan_outcome.rejection_reason
                        if result is None
                        else result.status,
                        "source_age_seconds": now
                        - plan_outcome.sensor_timestamp_seconds,
                        **planner_result_diagnostic(result),
                    }
                )
                if accepted:
                    active_plan = result
                    active_plan_lease = SafeSAGEPlanExecutionLease(
                        plan_outcome.sensor_timestamp_seconds,
                        maximum_candidate_age_seconds=args.planner_result_max_age,
                        execution_interval_seconds=mpc.config.time_step,
                        revalidation_interval_seconds=0.75 * mpc.config.time_step,
                    )
                    # Always revalidate delayed results at receipt, including
                    # state drift and any field changes during the solve.
                    warm = control.shifted_warm_start(result)
                else:
                    active_plan = None
                    active_plan_lease = None
                    if args.immediate_replan:
                        next_plan_due = now
            status = buffer.inspect(current_simulation_time_seconds=now)
            fresh = not status.fail_closed
            if fresh and args.experiment_baseline != "geometric_velocity_cbf" and not planner.busy and now >= next_plan_due:
                reuse = buffer.reuse(current_simulation_time_seconds=now)
                fresh = not reuse.fail_closed
                next_plan_due = now + 0.75 * mpc.config.time_step
            else:
                reuse = None
            if fresh and reuse is not None:
                horizon = waypoint.predict_commands(
                    RobotPose2D(*current),
                    horizon_steps=mpc.config.horizon_steps,
                    time_step=mpc.config.time_step,
                )
                sequence = np.array(
                    [[item.linear_velocity, item.angular_velocity] for item in horizon]
                )
                planner.submit(
                    frame_id=plan_id,
                    sensor_timestamp_seconds=now,
                    packet=(current.copy(), sequence, latest.field, warm,
                        np.array(ROUTE[waypoint.current_waypoint_index], dtype=float) if args.guidance_tracking_weight > 0 else None),
                )
                plan_waypoints[plan_id] = waypoint.current_waypoint_index
                plan_id += 1
            stop_reason = None
            command = np.zeros(2)
            geometric_diagnostic = None
            if args.experiment_baseline == "geometric_velocity_cbf":
                from robot.evaluation.restaurant_baselines import geometric_cbf_command
                if fresh:
                    command, geometric_diagnostic = geometric_cbf_command(
                        latest.field, latest.source_occupancy_grid, current,
                        (nominal.linear_velocity, nominal.angular_velocity), footprint, now,
                        alpha=stationary.CBF_GAMMA,
                    )
                    stop_reason = geometric_diagnostic["stop_reason"]
                    if geometric_diagnostic["cbf_margin"] is not None and stop_reason is None:
                        minimum_cbf_margin = min(minimum_cbf_margin, geometric_diagnostic["cbf_margin"])
                else:
                    stop_reason = "stale_field"
            elif not fresh:
                stop_reason = "stale_field"
            elif active_plan is None or not active_plan_lease.candidate_available(now):
                stop_reason = "waiting_for_fresh_valid_plan"
            else:
                # Re-evaluate the complete remaining horizon from the actual
                # state against each newly accepted field, with identical exact
                # DTCBF and dense geometry gates. This performs no new solve.
                if active_plan_lease.needs_validation(now, buffer.generation):
                    valid = mpc.revalidate_commands(
                        current,
                        active_plan.command_sequence,
                        configuration_space=latest.field.pipeline.configuration_space,
                        h_full=latest.field.pipeline.h_full,
                        guidance_fields=latest.field.pipeline.guidance_fields,
                        temporal_safety_derivative=latest.field.temporal_safety_derivative,
                    )
                    active_plan_lease.record_validation(
                        now, buffer.generation, accepted=valid
                    )
                    if not valid:
                        active_plan = None
                        stop_reason = "new_field_invalidated_plan"
                        if args.immediate_replan and not planner.busy:
                            next_plan_due = now
                if active_plan is not None and active_plan_lease.can_execute(
                    now, buffer.generation, field_fresh=fresh
                ):
                    filtered = control.filter_command(
                        (active_plan.linear_velocity, active_plan.angular_velocity),
                        current,
                        latest.field,
                    )
                    if (
                        filtered.constraint_feasible
                        and filtered.constraint_satisfied
                        and not filtered.emergency_stop_active
                        and math.isfinite(filtered.cbf_margin)
                    ):
                        command = np.array(
                            [
                                filtered.safe_linear_velocity,
                                filtered.safe_angular_velocity,
                            ]
                        )
                        minimum_cbf_margin = min(
                            minimum_cbf_margin, filtered.cbf_margin
                        )
                    else:
                        stop_reason = "cbf_fail_closed"
                        active_plan = None
                elif active_plan is not None:
                    stop_reason = "unverified_command_lease"
            if stop_reason:
                stops.append({"time": now - start_sim, "reason": stop_reason})
            trajectory.append(current.tolist())
            commands.append(
                {
                    "time": now - start_sim,
                    "v": float(command[0]),
                    "omega": float(command[1]),
                    "field_age": status.field_age_seconds,
                    "generation": buffer.generation,
                    "target_waypoint_index": waypoint.current_waypoint_index,
                    "stop_reason": stop_reason,
                    "nominal_v": nominal.linear_velocity,
                    "nominal_omega": nominal.angular_velocity,
                    "experiment_baseline": args.experiment_baseline,
                    "geometric_cbf": geometric_diagnostic,
                    "optimizer_candidate_source_age": None
                    if active_plan_lease is None
                    else now - active_plan_lease.source_timestamp_seconds,
                    "horizon_validation_age": None
                    if active_plan_lease is None
                    or active_plan_lease.validated_timestamp_seconds is None
                    else now - active_plan_lease.validated_timestamp_seconds,
                }
            )
            step(command, paced=True)
    finally:
        controller.stop()
        if human is not None and not arrived:
            human.abort()
        inference.close()
        planner.close()
    minimum_clearance_before_stop = minimum_clearance
    stopping_sweep = sample_stopping_sweep(state_reader=state, step=step,
        clearance_reader=independent_clearance, frame_count=60)
    save_json("stopping-sweep.json", stopping_sweep)
    final_state = np.array(stopping_sweep["final_state"])
    stop_drift = stopping_sweep["translation_drift_m"]
    minimum_clearance = min(minimum_clearance, stopping_sweep["minimum_clearance_m"])
    if human is not None and human.carry is not None:
        minimum_clearance = min(minimum_clearance, min(
            r["independent_robot_clearance_m"] for r in human.records
            if "independent_robot_clearance_m" in r))
    moving_field_count = sum(
        np.linalg.norm(
            np.asarray(live_fields[i]["state_at_commit"][:2])
            - np.asarray(live_fields[i - 1]["state_at_commit"][:2])
        )
        > 0.001
        for i in range(1, len(live_fields))
    )
    completed = (
        arrived
        and minimum_clearance > 0.02
        and stop_drift < 0.05
        and len(live_fields) >= 10
        and moving_field_count >= 10
    )
    human_validation = None
    passing_validation = None
    if human is not None:
        from robot.evaluation.human_navigation_validation import summarize_human_navigation
        human_validation = summarize_human_navigation(live_fields, human.records,
            mode=args.restaurant_human, upper_continuation=args.human_upper_continuation,
            crossing_direction=args.human_crossing_direction)
        completed = completed and human_validation["passed"]
        if args.human_body_envelope_radius is not None:
            from robot.evaluation.human_body_envelope_validation import validate_human_body_envelopes
            envelope_validation = validate_human_body_envelopes(live_fields, human.records,
                reference_radius_m=human.setup["collision_radius_m"])
            save_json("live-human-envelope-validation.json", envelope_validation)
            human_validation["human_envelope_checks"] = envelope_validation["checks"]
            completed = completed and envelope_validation["passed"]
        save_json("human-navigation-validation.json", human_validation)
        save_json("human-truth-evaluation-only.json", human.records)
        if args.motion_aware_passing:
            from robot.evaluation.motion_passing_validation import summarize_motion_passing
            passing_goal = (anchor.x + math.cos(anchor.yaw)*ROUTE[-1][0] - math.sin(anchor.yaw)*ROUTE[-1][1],
                            anchor.y + math.sin(anchor.yaw)*ROUTE[-1][0] + math.cos(anchor.yaw)*ROUTE[-1][1])
            passing_validation = summarize_motion_passing(live_fields, human.records, moving_robot=True,
                                                         nominal_goal_world_xy=passing_goal)
            completed = completed and passing_validation["passed"]
            save_json("motion-passing-validation.json", passing_validation)
        if human.carry is not None:
            from robot.scene.restaurant_human_carry import summarize_carry_records
            carry_validation = summarize_carry_records(human.carry.records, continuous=human.circuit is not None)
            carry_validation["source_asset_unchanged"] = hashlib.sha256(Path(human.carry_setup["source"]).read_bytes()).hexdigest() == human.carry_setup["source_sha256"]
            carry_validation["scope"] = "native attachment during live robot navigation; separate bowl physical perception remains unvalidated; no actor endpoint-stop claim"
            completed = completed and carry_validation["passed"] and carry_validation["source_asset_unchanged"]
            save_json("carry-validation.json", carry_validation)
            save_json("carry-records.json", human.carry.records)
    simultaneous_kinetic_updates = 0
    live_kinetic_validation = None
    if args.moving_actor:
        for record in live_fields:
            monitor = record["actor_monitor"]
            if (abs(monitor["velocity_world_xy"][1]) <= 0
                or np.linalg.norm(record["diagnostics"]["measured_robot_velocity_world_xy"]) <= 0.02):
                continue
            matches = [item for item in record["objects"] if np.linalg.norm(
                np.array(item["diagnostics"]["centroid_world_xy"]) - monitor["center_world_xy"]) < 0.4]
            if (len(matches) == 1 and matches[0]["diagnostics"]["policy_return_count"] > 0
                and matches[0]["diagnostics"]["factor_scores"]["kinetic"]["mode"] == "evidence_supported"):
                simultaneous_kinetic_updates += 1
        live_kinetic_validation = summarize_live_relative_motion(live_fields)
        completed = completed and simultaneous_kinetic_updates >= 5 and live_kinetic_validation["passed"]
    scene_unchanged = (
        original_hash
        == hashlib.sha256(SCENE_PATH.read_bytes()).hexdigest()
    ) and all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in source_hashes.items())
    save_json("live-fields.json", live_fields[-5:] if args.stream_live_records else live_fields)
    save_json("captures.json", captures)
    save_json("commands.json", commands)
    save_json("trajectory.json", trajectory)
    save_json("waypoint-events.json", waypoint_events)
    save_json("nearby-track-coverage-events.json", coverage_events)
    report = {
        "acceptance_scope": "bounded_navigation_with_explicit_factor_and_property_priors; all-track numeric scores are separate from physical perception readiness"
        if args.persistent_object_risk_map
        else "bounded_conservative_navigation; four-factor availability reported separately",
        "passed": completed and scene_unchanged and root_unchanged_by_fixture,
        "experiment_baseline": args.experiment_baseline,
        "human_body_contact_prior_config": asdict(perception.human_body_contact_config) if perception.human_body_contact_config is not None else None,
        "human_body_envelope_config": asdict(perception.human_body_envelope.config) if perception.human_body_envelope is not None else None,
        "human_navigation_validation": human_validation,
        "motion_passing_validation": passing_validation,
        "native_human_mode": args.restaurant_human,
        "human_lidar_cluster_radius_m": args.human_cluster_radius,
        "native_carried_bowl": args.human_carry_bowl,
        "native_human_upper_continuation": args.human_upper_continuation,
        "human_crossing_trigger_robot_world_y": args.human_crossing_trigger_y,
        "human_crossing_delay_seconds": args.human_crossing_delay_seconds,
        "human_crossing_direction": args.human_crossing_direction,
        "motion_aware_passing_config": asdict(perception.motion_passing.config) if perception.motion_passing is not None else None,
        "actor_trigger_basis": "exogenous_navigation_elapsed_time" if args.human_crossing_delay_seconds is not None else "robot_world_y",
        "human_requested_walking_speed_mps": args.human_walking_speed,
        "experimental_camera_mount": {
            "upward_degrees": args.human_camera_upward_degrees,
            "raise_metres": args.human_camera_raise_metres,
        },
        "human_mass_prior_config": asdict(perception.human_mass_prior_config) if human is not None else None,
        "scene_source_sha256": source_hashes,
        "completed": arrived,
        "reason": reason,
        "minimum_clearance_m": minimum_clearance,
        "minimum_clearance_before_stop_m": minimum_clearance_before_stop,
        "minimum_clearance_during_stop_m": stopping_sweep["minimum_clearance_m"],
        "stop_yaw_drift_radians": stopping_sweep["yaw_drift_radians"],
        "minimum_cbf_margin": None
        if not math.isfinite(minimum_cbf_margin)
        else minimum_cbf_margin,
        "stop_drift_m": stop_drift,
        "last_state": final_state.tolist(),
        "live_field_count": len(live_fields),
        "live_field_record_storage": "full history in live-fields.jsonl; live-fields.json is latest-five checkpoint"
        if args.stream_live_records else "full history in live-fields.json",
        "fields_committed_while_translating": moving_field_count,
        "simultaneous_robot_actor_kinetic_updates": simultaneous_kinetic_updates,
        "live_relative_motion_validation": live_kinetic_validation,
        "nearby_track_loss_guard": coverage_result,
        "nearby_track_loss_guard_config": asdict(coverage_guard.config) if coverage_guard is not None else None,
        "perception_evidence_counts": {
            "accepted_current_instances": sum(
                r["diagnostics"].get("accepted_instance", False) for r in live_fields
            ),
            "four_factor_ready_updates": sum(
                r["readiness"]["ready"] for r in live_fields
            ),
            "ready_updates_by_factor": {
                name: sum(
                    next(
                        f["accepted"]
                        for f in r["readiness"]["factors"]
                        if f["factor"] == name
                    )
                    for r in live_fields
                )
                for name in ("potential", "kinetic", "contact", "thermal")
            },
            "retained_updates_by_property": {
                name: sum(
                    r["diagnostics"]
                    .get("temporal_object_evidence", {})
                    .get(name, {})
                    .get("mode")
                    in {"retained", "retained_stronger_contact"}
                    for r in live_fields
                )
                for name in ("mass", "contact")
            },
        },
        "all_object_policy_counts": {
            "object_assessments": sum(len(r["objects"]) for r in live_fields),
            "accepted_visual_instances": sum(
                r["diagnostics"].get("accepted_instance_count", 0) for r in live_fields
            ),
            "perception_ready_assessments": sum(
                r["diagnostics"].get("complete_perception_count", 0)
                for r in live_fields
            ),
            "configured_factor_prior_uses": sum(
                value["mode"] == "configured_prior"
                for r in live_fields
                for item in r["objects"]
                for value in item["diagnostics"]["factor_scores"].values()
            ),
            "persistent_property_uses": sum(
                value["mode"] == "persistence_prior"
                for r in live_fields
                for item in r["objects"]
                for value in item["diagnostics"]["property_evidence"].values()
            ),
        },
        "object_factor_prior_risk": args.object_factor_prior_risk
        if args.persistent_object_risk_map
        else None,
        "persistent_property_config": asdict(perception.persistent_properties.config)
        if args.persistent_object_risk_map
        else None,
        "temporal_evidence_config": None
        if perception.evidence_memory is None
        else asdict(perception.evidence_memory.config),
        "safety_stops": stops,
        "plan_events": plan_events,
        "rejections": sorted(set(rejections)),
        "scene_unchanged": scene_unchanged,
        "simulation_rate_limit": args.simulation_rate,
        "maximum_simulation_seconds": args.maximum_seconds,
        "planner_candidate_maximum_age_seconds": args.planner_result_max_age,
        "recovery_lateral_hint_grid_y": None,
        "recovery_cost_pruning": mpc.config.enable_recovery_cost_pruning,
        "recovery_feasibility_pruning": mpc.config.enable_recovery_feasibility_pruning,
        "qp_iteration_limit": mpc.config.maximum_solver_iterations,
        "nonlinear_rescue_iteration_limit": mpc.config.maximum_nonlinear_rescue_iterations,
        "progress_seed_linear_fraction": mpc.config.progress_seed_linear_fraction,
        "mpc_configuration": asdict(mpc.config),
        "guidance_tracking_ablation": {"enabled": args.guidance_tracking_weight > 0,
            "weight": args.guidance_tracking_weight, "activation_safety_value": args.guidance_activation_h,
            "maximum_heading_bias_radians": math.radians(args.guidance_max_heading_bias_degrees),
            "reference_goal_radius_m": args.guidance_reference_goal_radius,
            "guidance_fade_distance_m": args.guidance_fade_distance,
            "minimum_reference_progress_m": args.guidance_minimum_reference_progress,
            "minimum_turn_first_rate_cap": args.guidance_turn_first_minimum_cap,
            "reference_mode": args.guidance_reference_mode,
            "forward_speed_reference": "maximum forward speed in original obstacle-unaware nominal horizon, clipped to actuator bounds",
            "heading_reference_basis": "active_obstacle_unaware_waypoint; same field frame; request-time waypoint index",
            "scope": "separate checked local guidance-path tracking objective; primary nominal unchanged; not paper equivalence"},
        "immediate_replan_after_invalidations": args.immediate_replan,
        "verified_command_interval_seconds": mpc.config.time_step,
        "wall_seconds": time.perf_counter() - start_wall,
        "navigation_simulation_seconds": now - start_sim,
        "controller_scope": "circular_velocity_aware_cbf_nominal_omega_no_mpc; shadow_CRA_fields_for_matched_perception" if args.experiment_baseline == "geometric_velocity_cbf" else "checked_guidance_local_path_ablation_with_exact_DTCBF_geometry_and_analytical_CBF" if args.guidance_tracking_weight > 0 else "online_unicycle_MPC_exact_DTCBF_geometry_and_analytical_CBF",
        "fixed_lambda": 1.25,
        "plume_control": args.plume_control,
        "restaurant_scene": str(SCENE_PATH) if args.restaurant_scene else None,
        "static_restaurant_server": args.restaurant_server,
        "sam_observed_context_box_retry": args.sam_observed_context_box,
        "server_component_gate": gate.get("server_component_checks"),
        "restaurant_component_gate": gate.get("restaurant_component_checks"),
        "independent_authored_layout_monitor": bool(layout_monitor is not None),
        "rgb_output_resolution_height_width": [calibration.output_height, calibration.output_width],
        "fixture_layout": args.fixture_layout,
        "moving_actor": args.moving_actor,
        "actor_shape": "real_restaurant_meshes" if args.restaurant_scene else args.actor_shape,
        "plume_view_fov_degrees": args.plume_view_fov,
        "goal_local_xy": GOAL,
        "nominal_route_local_xy": ROUTE,
        "restaurant_route": args.restaurant_route if args.restaurant_scene else None,
        "waypoints_reached": len(ROUTE) if arrived else waypoint.current_waypoint_index,
        "fixture_spacing": args.fixture_spacing,
        "progress_timeout_seconds": args.progress_timeout_seconds,
        "clearance_monitor_scope": "independent authored original mesh raster plus conservative new-prop XY boxes and selected bowl disk; base footprint only; no payload; never control/factor input"
        if args.restaurant_scene else "independent authored sphere disks and circumscribed support disk; never control/factor input",
        "scheduled_observation_pauses": 0,
        "lidar_ego_motion_correction": "legacy_scan_start_transform"
        if args.disable_lidar_deskew
        else "per_return_ego_motion_deskew",
        "rtx_history_setting_requested": carb.settings.get_settings().get(
            "/rtx/scenedb/maxHistoryTransformCount"
        ),
        "boundaries": [
            "paced simulation; not real-time",
            "RGB read time approximates acquisition; per-return ego deskew uses oracle pose history and corrects only sensor motion"
            if not args.disable_lidar_deskew
            else "RGB read time approximates acquisition; legacy scans are not per-return deskewed",
            "oracle USD ego poses/extrinsics and fixture clearance monitor",
            "ground/fall interaction priors remain explicit",
            "session-persistent mass, visible contact, support and initial plume are explicit unchanged-property priors; original timestamps retained; motion/route interaction remain live"
            if args.persistent_object_risk_map
            else "short-lived retained mass/contact are persistence priors with original timestamps; no cached motion/plume/fall-height evidence",
            "renderer internal history warning remains unresolved despite the requested setting",
            "SAM-associated category-free rough heaviness, visual contact silhouette and registered visible-plume proxies",
            "new restaurant assets have no emitted plume or validated negative-plume scope; thermal remains a labeled factor prior; all live LiDAR occupancy retained"
            if args.restaurant_scene else "rendered puffs are opaque visual fixtures, not steam physics/temperature; all live LiDAR occupancy retained",
            "real-mesh restaurant component gate; potential/support and thermal may remain configured priors; not full four-factor sensor acceptance, moving-view plume or human validation"
            if args.restaurant_scene else "bounded separated sources with validated initial clean-reference registration and retained cues; no moving-view plume/human/crowd generalization",
        ],
    }
    if human is not None:
        report["limitations"] = [
            "one native adult; human mass is configured 70 [45,100] kg prior selected by fresh person/SAM association; original catalog estimates retained",
            "only intrinsic mass persists for at most 3 original-source seconds; current view contact, support, plume, motion and old body bounds never borrowed",
            "potential/thermal use explicit factor priors where unavailable; visual silhouette is not physical sharpness, hardness or injury severity",
            "all physical LiDAR occupancy retained with .15 s positive expiry; track-loss watchdog is bounded, not general occlusion or crowd support",
            "oracle ego pose/extrinsics/gravity; RGB acquisition is read-time approximation; ego deskew does not correct moving articulated actors",
            ("geometric circular velocity-aware CBF reference; nominal omega retained, no MPC/DTCBF/h_full command; shadow fields retain lambda 1.25"
             if args.experiment_baseline == "geometric_velocity_cbf" else
             "fixed lambda 1.25, unchanged nominal/exact MPC/DTCBF/analytical CBF; paced development, not real-time or causal CRA superiority"),
            "renderer transform-history warning remains unresolved; separate carried-bowl physical assessment, calibrated load and moving-view plume remain unvalidated",
        ]
    save_json("report.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "completed",
                    "reason",
                    "minimum_clearance_m",
                    "live_field_count",
                    "fields_committed_while_translating",
                    "stop_drift_m",
                )
            },
            default=json_default,
        ),
        flush=True,
    )
    timeline.stop()
    if not report["passed"]:
        raise RuntimeError("live wheel gate failed; inspect saved diagnostics")


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    if ACTIVE_HUMAN is not None:
        ACTIVE_HUMAN.abort()
        save_json("human-truth-evaluation-only.json", ACTIVE_HUMAN.records)
        if ACTIVE_HUMAN.carry is not None:
            save_json("carry-records.json", ACTIVE_HUMAN.carry.records)
            save_json("native-carry-setup.json", ACTIVE_HUMAN.carry_setup)
        if ACTIVE_HUMAN.circuit is not None:
            save_json("continuous-server-route-events.json", ACTIVE_HUMAN.circuit.events)
        ACTIVE_HUMAN.close()
    app.close()
