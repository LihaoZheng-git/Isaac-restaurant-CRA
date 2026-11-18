"""Fixed outbound service fixture. Authored hazards stay outside control inputs."""

import math

from robot.evaluation.restaurant_right_route import PICKUP
from robot.evaluation.service_crate_short_probe import REFERENCE_RECIPE


SCENE_NAME = "restaurant_left_outbound.usda"
LOUNGE_BIN_SCENE_NAME = "restaurant_left_outbound_lounge_bin.usda"
START_WORLD_XY_YAW = (25.80, 6.25, -math.pi / 2)
LEFT_PICKUP = (*PICKUP[:2], math.pi)
WEST_PICKUP = (23.30, 4.00, math.pi)
GUIDES_WORLD_XY = (
    (25.80, 4.85), (24.45, 4.05), (5.80, 4.20), (2.65, 7.15),
    (2.35, 8.20), (2.35, 13.40), (3.80, 14.60), (5.55, 15.95),
)
WIDE_EXIT_GUIDES_WORLD_XY = ((26.15, 4.10), *GUIDES_WORLD_XY[1:])
FIELD_ALIGNMENT_GUIDES_WORLD_XY = ((25.85, 4.30), *GUIDES_WORLD_XY[1:])
BIN_CENTER_XY = (17.80, 3.50)
BIN_VIEW_WORLD_XY_YAW = (19.65, 4.09, math.pi)
LOUNGE_BIN_CENTER_XY = (9.25, 4.70)
LOUNGE_BIN_VIEW_WORLD_XY_YAW = (11.10, 4.16, math.pi)
MAXIMUM_DOCK_OBSERVATION_FRAMES = 90
DOCK_PROBE_MAXIMUM_SECONDS = 35.0
DOCK_PROBE_PROGRESS_TIMEOUT_SECONDS = 12.0
AISLE_PROBE_START_XY = (24.30, 4.05 + (24.45 - 24.30) * .15 / 18.65)
AISLE_PROBE_START_YAW = math.atan2(.15, -18.65)
AISLE_PROBE_ROUTE = ((4., 0.),)
AISLE_PROBE_MAXIMUM_SECONDS = 35.
AISLE_PROBE_STALL_SECONDS = 5.
AISLE_PROBE_LATERAL_LIMIT_RADIANS = math.radians(12.)
AISLE_BIN_START_XY = (26.15, 4.05 + (24.45 - 26.15) * .15 / 18.65)
AISLE_BIN_APPROACH_WORLD_X = 11.10
AISLE_BIN_FINISH_WORLD_X = 7.40
AISLE_BIN_MAXIMUM_SECONDS = 125.
HUMAN_PROBE_START_WORLD_XY_YAW = (1.90, 8.00, math.pi / 2)
HUMAN_PROBE_GOAL_WORLD_XY = (1.90, 13.40)
HUMAN_PROBE_MAXIMUM_SECONDS = 60.
FULL_COMMITTED_GUIDE_INDEX = 2
FULL_V5_START_GUIDE_INDEX = 3
FULL_ROUTE_HUMAN_LANE_X = 2.05
FULL_ROUTE_COMMITTED_GUIDES_WORLD_XY = (
    *FIELD_ALIGNMENT_GUIDES_WORLD_XY[:4],
    (FULL_ROUTE_HUMAN_LANE_X, HUMAN_PROBE_START_WORLD_XY_YAW[1]),
    (FULL_ROUTE_HUMAN_LANE_X, HUMAN_PROBE_GOAL_WORLD_XY[1]),
    *FIELD_ALIGNMENT_GUIDES_WORLD_XY[6:],
)
FULL_ROUTE_LOUNGE_GUIDES_WORLD_XY = (
    *FULL_ROUTE_COMMITTED_GUIDES_WORLD_XY[:3],
    (1.65, 5.65),
    *FULL_ROUTE_COMMITTED_GUIDES_WORLD_XY[4:-1],
    (4.50, 15.00),
)
HUMAN_TURN_PROBE_START_WORLD_XY_YAW = (5.7934, 4.7070, math.radians(150.13))
HUMAN_TURN_PROBE_MAXIMUM_SECONDS = 90.
LIDAR_DELIVERY_START_WORLD_XY_YAW = (7.00, 4.30, math.pi)
LIDAR_DELIVERY_GUIDES_WORLD_XY = FULL_ROUTE_LOUNGE_GUIDES_WORLD_XY[2:]
LIDAR_DELIVERY_DISABLED_PRIORS = (
    "human_track_recovery", "human_tracking_uncertainty", "human_verified_part_recovery",
    "human_prior_geometry_associations", "human_fragment_coverage_prior",
    "human_short_fragment_coverage", "stationary_furniture_memory",
    "stationary_furniture_fragment_proof", "unknown_obstacle_planning",
    "unknown_supported_background_prior", "unknown_persistent_background_prior",
    "unknown_unclassified_static_prior", "unknown_verified_dynamic_evidence",
    "unknown_distinct_surface_witnesses",
)
WAYPOINT_PRIORITY_MINIMUM_GUIDE_PROJECTION = .30
WAYPOINT_PRIORITY_NEAR_SCHEDULED_FRAMES = 24


def _guides_world(*, wide_exit, field_alignment):
    if field_alignment:
        return FIELD_ALIGNMENT_GUIDES_WORLD_XY
    return WIDE_EXIT_GUIDES_WORLD_XY if wide_exit else GUIDES_WORLD_XY


def _local_points(start, points):
    x, y, yaw = start
    c, s = math.cos(yaw), math.sin(yaw)
    return tuple(((wx-x)*c + (wy-y)*s, -(wx-x)*s + (wy-y)*c) for wx, wy in points)


def nominal_route_local(start=START_WORLD_XY_YAW, *, wide_exit=False, field_alignment=False,
                        full_route_committed_bypass=False):
    guides = (FULL_ROUTE_COMMITTED_GUIDES_WORLD_XY if full_route_committed_bypass
              else _guides_world(wide_exit=wide_exit, field_alignment=field_alignment))
    return _local_points(start, guides)


def validate_left_outbound(args, reference_args, reference_case):
    """Require V5 settings and audit the explicit objective and probe selectors."""
    field_alignment = getattr(args, "left_outbound_field_alignment", False)
    dock_probe = getattr(args, "left_outbound_dock_probe", False)
    aisle_probe = getattr(args, "left_outbound_aisle_probe", False)
    aisle_bin = getattr(args, "left_outbound_aisle_bin_probe", False)
    human_probe = getattr(args, "left_outbound_human_probe", False)
    human_turn_probe = getattr(args, "left_outbound_human_turn_probe", False)
    lidar_delivery = getattr(args, "left_outbound_lidar_delivery_probe", False)
    lidar_full_delivery = getattr(args, "left_outbound_lidar_full_delivery", False)
    common_pickup = getattr(args, "left_outbound_common_pickup", False)
    preloaded_west = getattr(args, "left_outbound_preloaded_west", False)
    lidar_guided_prop_masks = getattr(args, "prop_masks_lidar_guided_only", False)
    lidar_occupancy = lidar_delivery or lidar_full_delivery
    west_diner = getattr(args, "left_outbound_west_diner", False)
    full_route_bypass = getattr(args, "left_outbound_full_route_committed_bypass", False)
    lounge_walk = getattr(args, "left_outbound_lounge_walk", False)
    no_laplace_objective = getattr(args, "left_outbound_no_laplace_objective", False)
    committed_bypass = getattr(args, "left_outbound_committed_bypass", False)
    nominal_preserving_bypass = getattr(args, "left_outbound_nominal_preserving_bypass", False)
    waypoint_priority = getattr(args, "left_outbound_waypoint_priority", False)
    priority_near = waypoint_priority and reference_case == "long-parallel-near"
    if (not args.restaurant_scene or args.restaurant_scene.name not in {SCENE_NAME, LOUNGE_BIN_SCENE_NAME}
            or args.service_crate_short_probe or args.service_lounge_bin_probe
            or args.service_parked_trolley_probe
            or (args.left_outbound_bin_view and (args.motion or reference_case != "long-parallel"))
            or (reference_case == "long-parallel-near" and args.motion)):
        raise ValueError("left outbound requires its separate stage and zero-wheel near/bin views")
    if field_alignment and (not args.left_outbound_wide_exit or args.left_outbound_bin_view
                            or reference_case != "long-parallel"):
        raise ValueError("left outbound field alignment requires the wide-exit delivery case")
    if dock_probe and not field_alignment:
        raise ValueError("left outbound dock probe requires direct field alignment")
    if human_probe and (
        reference_case != "long-parallel" or args.restaurant_scene.name != LOUNGE_BIN_SCENE_NAME
        or not args.long_service_parallel or not args.straight_service_walk
        or args.maximum_seconds != (HUMAN_TURN_PROBE_MAXIMUM_SECONDS if human_turn_probe
                                    else HUMAN_PROBE_MAXIMUM_SECONDS)
        or args.left_outbound_bin_view or args.left_outbound_wide_exit or args.left_outbound_cast_manifest
        or field_alignment or dock_probe or waypoint_priority or aisle_probe
    ):
        raise ValueError("left human probe requires its fixed-duration V5 oncoming walk on the west-lounge stage")
    if full_route_bypass and (
        reference_case != "long-parallel" or args.restaurant_scene.name != LOUNGE_BIN_SCENE_NAME
        or not (field_alignment and args.left_outbound_wide_exit and waypoint_priority
                and args.left_outbound_cast_manifest and args.long_service_parallel and args.straight_service_walk)
        or args.left_outbound_bin_view or dock_probe or aisle_probe or human_probe
    ):
        raise ValueError("full-route committed bypass requires the staged left delivery and original actor gates")
    if human_turn_probe and (not human_probe or not lounge_walk
                            or getattr(args, "left_outbound_human_lane_x", None) is not None):
        raise ValueError("the turn probe requires its fixed left human lane and lounge walk")
    if ((lidar_delivery and not human_turn_probe)
            or (lidar_full_delivery and not (full_route_bypass and lounge_walk))):
        raise ValueError("current LiDAR delivery requires its corresponding short or full route")
    if common_pickup and not (lidar_full_delivery and args.motion):
        raise ValueError("the common pickup requires a driven full left LiDAR delivery")
    if preloaded_west and (common_pickup or not (lidar_full_delivery and args.motion)):
        raise ValueError("the preloaded west pickup requires its own driven full left LiDAR delivery")
    if lidar_occupancy and (not west_diner or not args.human_observed_occupancy_only
                           or args.static_occupancy_memory_prior
                           or any(getattr(args, name) for name in LIDAR_DELIVERY_DISABLED_PRIORS)):
        raise ValueError("the LiDAR delivery requires the west diner and current LiDAR occupancy")
    if lounge_walk and not (full_route_bypass or human_turn_probe):
        raise ValueError("the longer lounge walk requires the full delivery or its left-turn probe")
    if aisle_bin and not aisle_probe:
        raise ValueError("aisle bin probe requires the static aisle selector")
    if no_laplace_objective and not aisle_bin:
        raise ValueError("waypoint-only objective requires the full aisle-and-bin probe")
    if committed_bypass and not (aisle_bin and no_laplace_objective):
        raise ValueError("committed bypass requires the waypoint-only full aisle-and-bin probe")
    if nominal_preserving_bypass and not (committed_bypass and aisle_bin and no_laplace_objective):
        raise ValueError("nominal-preserving bypass requires the committed waypoint-only aisle-and-bin probe")
    aisle_limit = AISLE_BIN_MAXIMUM_SECONDS if aisle_bin else AISLE_PROBE_MAXIMUM_SECONDS
    if aisle_probe and (args.restaurant_scene.name != LOUNGE_BIN_SCENE_NAME or not field_alignment
                        or not waypoint_priority or dock_probe or args.left_outbound_cast_manifest
                        or args.maximum_seconds != aisle_limit):
        raise ValueError("aisle probe requires its west-lounge stage, waypoint-priority alignment and fixed duration")
    if waypoint_priority and (not args.left_outbound_wide_exit or dock_probe
                              or not (field_alignment or priority_near)):
        raise ValueError("left outbound waypoint priority requires full direct field alignment or its stationary near-view gate")
    allowed = {"output", "restaurant_scene", "restaurant_layout_survey", "left_outbound_service",
               "left_outbound_bin_view", "left_outbound_cast_manifest", "left_outbound_wide_exit",
               "left_outbound_field_alignment", "left_outbound_dock_probe", "left_outbound_waypoint_priority",
               "motion", "long_service_human_gates"}
    if aisle_probe:
        allowed.update(("left_outbound_aisle_probe", "left_outbound_aisle_bin_probe",
                        "left_outbound_no_laplace_objective", "left_outbound_committed_bypass",
                        "left_outbound_nominal_preserving_bypass",
                        "maximum_seconds"))
    if human_probe:
        allowed.update(("left_outbound_human_probe", "left_outbound_human_lane_x", "maximum_seconds"))
    if human_turn_probe:
        allowed.update(("left_outbound_human_turn_probe", "left_outbound_lounge_walk"))
    if lidar_occupancy:
        allowed.update((*LIDAR_DELIVERY_DISABLED_PRIORS, "human_observed_occupancy_only",
                        "left_outbound_lidar_delivery_probe", "left_outbound_lidar_full_delivery"))
    if west_diner and (lidar_occupancy or reference_case == "long-parallel-near"):
        allowed.add("left_outbound_west_diner")
    if full_route_bypass:
        allowed.update(("left_outbound_full_route_committed_bypass", "left_outbound_lounge_walk"))
    if common_pickup:
        allowed.add("left_outbound_common_pickup")
    if preloaded_west:
        allowed.add("left_outbound_preloaded_west")
    if getattr(args, "restaurant_background_motion", False):
        if not (lidar_full_delivery and preloaded_west and args.motion):
            raise ValueError("Continuous restaurant gestures require the full preloaded west delivery")
        allowed.add("restaurant_background_motion")
    if lidar_guided_prop_masks:
        if not (lidar_full_delivery and args.stationary_image_objects):
            raise ValueError("the left-route prop mask trial requires the full LiDAR delivery")
        allowed.add("prop_masks_lidar_guided_only")
    actual, expected = vars(args), vars(reference_args)
    changed = sorted(key for key in actual.keys() | expected.keys()
                     if key not in allowed and actual.get(key) != expected.get(key))
    if changed:
        raise ValueError("left outbound changes the full service policy: " + ", ".join(changed))
    profile = dict(passed=True, reference_recipe=REFERENCE_RECIPE, reference_case=reference_case,
                   start_world_xy_yaw=START_WORLD_XY_YAW, bin_view=args.left_outbound_bin_view,
                   nominal_guides_world_xy=_guides_world(wide_exit=args.left_outbound_wide_exit,
                                                         field_alignment=field_alignment),
                   controller_perception_uncertainty_arguments_match=not field_alignment,
                   scope="live outbound food service; one bin and one native oncoming adult")
    if args.restaurant_scene.name == LOUNGE_BIN_SCENE_NAME:
        profile.update(bin_center_world_xy=LOUNGE_BIN_CENTER_XY,
                       bin_view_world_xy_yaw=LOUNGE_BIN_VIEW_WORLD_XY_YAW)
    if field_alignment:
        profile.update(controller_objective="direct_laplace_field_alignment",
                       perception_uncertainty_arguments_match=True)
    if dock_probe or (waypoint_priority and field_alignment):
        profile["minimum_guide_projection"] = WAYPOINT_PRIORITY_MINIMUM_GUIDE_PROJECTION
    if priority_near:
        profile.update(scope="stationary near-view prerequisite with bounded live scheduling",
                       waypoint_priority_near=True,
                       waypoint_priority_near_scheduled_frames=WAYPOINT_PRIORITY_NEAR_SCHEDULED_FRAMES)
    elif waypoint_priority:
        profile["waypoint_priority"] = True
    if dock_probe:
        profile.update(scope="bounded departure from the service bar through the first guide",
                       dock_probe=dict(maximum_seconds=DOCK_PROBE_MAXIMUM_SECONDS,
                                       progress_timeout_seconds=DOCK_PROBE_PROGRESS_TIMEOUT_SECONDS,
                                       required_guides=1))
    if human_probe and not human_turn_probe:
        distance = HUMAN_PROBE_GOAL_WORLD_XY[1] - HUMAN_PROBE_START_WORLD_XY_YAW[1]
        lane_x = getattr(args, "left_outbound_human_lane_x", None)
        lane_x = HUMAN_PROBE_START_WORLD_XY_YAW[0] if lane_x is None else lane_x
        profile.update(scope="isolated left oncoming walk with a straight obstacle-unaware waypoint",
                       start_world_xy_yaw=(lane_x, *HUMAN_PROBE_START_WORLD_XY_YAW[1:]),
                       nominal_guides_world_xy=((lane_x, HUMAN_PROBE_GOAL_WORLD_XY[1]),),
                       nominal_route_local_xy=((distance, 0.),),
                       human_probe=dict(maximum_seconds=HUMAN_PROBE_MAXIMUM_SECONDS))
    if human_turn_probe:
        start = LIDAR_DELIVERY_START_WORLD_XY_YAW if lidar_delivery else HUMAN_TURN_PROBE_START_WORLD_XY_YAW
        guides = LIDAR_DELIVERY_GUIDES_WORLD_XY if lidar_delivery else FULL_ROUTE_LOUNGE_GUIDES_WORLD_XY[3:]
        previous = start[:2] if lidar_delivery else FULL_ROUTE_LOUNGE_GUIDES_WORLD_XY[2]
        profile.update(
            scope="isolated left turn, lounge-bound oncoming adult and full-route table approach",
            start_world_xy_yaw=start,
            nominal_guides_world_xy=guides,
            nominal_route_local_xy=_local_points(start, guides),
            previous_guide_world_xy=previous,
            previous_guide_local_xy=_local_points(start, (previous,))[0],
            human_probe=dict(maximum_seconds=HUMAN_TURN_PROBE_MAXIMUM_SECONDS,
                             progress_stall_seconds=15.0),
        )
        if lidar_delivery:
            profile.update(scope="preloaded short delivery with a west-side diner and current LiDAR occupancy",
                           current_lidar_occupancy_only=True, west_side_diner=True,
                           controller_perception_uncertainty_arguments_match=False)
    if full_route_bypass:
        guides = FULL_ROUTE_LOUNGE_GUIDES_WORLD_XY if lounge_walk else FULL_ROUTE_COMMITTED_GUIDES_WORLD_XY
        profile.update(
            scope="left delivery with the nominal-preserving westbound bin bypass and V5 oncoming walk",
            nominal_guides_world_xy=guides,
            nominal_route_local_xy=_local_points(START_WORLD_XY_YAW, guides),
            full_route_committed_bypass=dict(
                static_guide_index=FULL_COMMITTED_GUIDE_INDEX,
                v5_first_guide_index=FULL_V5_START_GUIDE_INDEX,
                nominal_preserving=True,
            ),
        )
        if lidar_full_delivery:
            anchor = LIDAR_DELIVERY_START_WORLD_XY_YAW
            start = LEFT_PICKUP if common_pickup else START_WORLD_XY_YAW
            guides = (*FULL_ROUTE_LOUNGE_GUIDES_WORLD_XY[:2], anchor[:2], *LIDAR_DELIVERY_GUIDES_WORLD_XY)
            profile.update(
                scope="bar pickup, committed bin bypass and the short current-LiDAR delivery route",
                start_world_xy_yaw=start,
                current_lidar_occupancy_only=True, west_side_diner=True,
                controller_perception_uncertainty_arguments_match=False,
                nominal_guides_world_xy=guides,
                nominal_route_local_xy=_local_points(anchor, guides),
                field_anchor_world_xy_yaw=anchor,
                previous_guide_local_xy=_local_points(anchor, (start[:2],))[0],
                short_route_first_guide_index=FULL_V5_START_GUIDE_INDEX,
            )
            if common_pickup:
                profile["common_pickup"] = True
            if preloaded_west:
                skipped_guides = FULL_COMMITTED_GUIDE_INDEX
                route = profile["full_route_committed_bypass"]
                route["static_guide_index"] -= skipped_guides
                route["v5_first_guide_index"] -= skipped_guides
                profile.update(
                    preloaded_west=True,
                    scope="preloaded west pickup, committed bin bypass and the current-LiDAR left delivery",
                    start_world_xy_yaw=WEST_PICKUP,
                    nominal_guides_world_xy=guides[skipped_guides:],
                    nominal_route_local_xy=_local_points(anchor, guides[skipped_guides:]),
                    previous_guide_local_xy=_local_points(anchor, (guides[skipped_guides - 1],))[0],
                    short_route_first_guide_index=route["v5_first_guide_index"],
                )
    if lounge_walk:
        from robot.scene.restaurant_straight_walk import (
            LEFT_OUTBOUND_LOUNGE_ACTOR_GOAL, LEFT_OUTBOUND_LOUNGE_TRIGGER_Y, LEFT_OUTBOUND_WEST_DINER_GOAL,
        )

        profile["lounge_walk"] = dict(actor_trigger_robot_world_y=LEFT_OUTBOUND_LOUNGE_TRIGGER_Y,
                                      actor_goal_world_xyz=LEFT_OUTBOUND_WEST_DINER_GOAL if west_diner
                                      else LEFT_OUTBOUND_LOUNGE_ACTOR_GOAL)
    if aisle_probe:
        start = AISLE_BIN_START_XY if aisle_bin else AISLE_PROBE_START_XY
        cosine, sine = math.cos(AISLE_PROBE_START_YAW), math.sin(AISLE_PROBE_START_YAW)
        route = (((AISLE_BIN_FINISH_WORLD_X-start[0]) / cosine, 0.),) if aisle_bin else AISLE_PROBE_ROUTE
        distance = route[-1][0]
        endpoint = (start[0] + distance * cosine, start[1] + distance * sine)
        scope = ("continuous aisle approach and conditional lounge-bin pass with a held remote diner"
                 if aisle_bin else "four-metre static aisle drive with the native diner held remotely")
        objective = ("waypoint_only_mpc_objective" if no_laplace_objective
                     else "signed_lateral_laplace_about_local_waypoint")
        profile.update(scope=scope, controller_objective=objective,
                       start_world_xy_yaw=(*start, AISLE_PROBE_START_YAW),
                       nominal_guides_world_xy=(endpoint,),
                       nominal_route_local_xy=route,
                       progress_stall_seconds=AISLE_PROBE_STALL_SECONDS,
                       lateral_guidance_limit_radians=0. if no_laplace_objective else AISLE_PROBE_LATERAL_LIMIT_RADIANS)
        if aisle_bin:
            profile['approach_checkpoint_local_x'] = (AISLE_BIN_APPROACH_WORLD_X-start[0]) / cosine
        if committed_bypass:
            profile['committed_bypass'] = {'mode': 'static_lounge_bin'}
            if nominal_preserving_bypass:
                profile['committed_bypass']['nominal_preserving'] = True
    return profile
