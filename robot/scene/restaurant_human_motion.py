"""Native Isaac human motion setup; authored actor state is evaluation truth.

Import the native helpers only after SimulationApp. This module never supplies
actor position, configured speed, mass, geometry or labels to CRA estimation.
"""

import hashlib

from pxr import Gf, UsdGeom, UsdPhysics, UsdSkel

from robot.scene.restaurant_server import SERVER_PRIM_PATH

HUMAN_MOTION_LIBRARY_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/People/MotionLibrary/HumanMotionLibrary.usd"
HUMAN_MOTION_LIBRARY_SHA256 = (
    "9a6f8b5df9426d609f29c48feace8933d636f8c7a9b15711b88313550bb49b8d"
)
HUMAN_MOTION_LIBRARY_PRIM = "/World/CRAServiceActors/HumanMotionLibrary"
HUMAN_WALK_GOAL = (29.8, 12.5, 0.0)


def human_navmesh_extent(*, open_layout=False, parallel_oncoming=False, upper_departure=False, straight_service_walk=False):
    """Actor authoring domain only; no robot planner or geometry input."""
    if (upper_departure or straight_service_walk) and not parallel_oncoming:
        raise ValueError('upper actor NavMesh requires the parallel encounter')
    low = (25., 6., -.2) if open_layout or parallel_oncoming else (27., 7., -.2)
    return low, (31., 25. if upper_departure or straight_service_walk else 14., 2.5)


def configure_human_navmesh(stage):
    """Explicit metre settings, limited to this process's unsaved stage root."""
    if (
        UsdGeom.GetStageMetersPerUnit(stage) != 1
        or UsdGeom.GetStageUpAxis(stage) != "Z"
    ):
        raise ValueError("bounded human motion requires a metre/Z-up stage")
    values = {
        "agentMinHeight": 1.7,
        "agentMinRadius": 0.2,
        "agentMaxRadius": 0.4,
        "agentMaxStepHeight": 0.2,
        "agentMaxFloorSlope": 20.0,
        "agentMinIslandRadius": 0.3,
        "agentSamplingDistance": 0.1,
        "excludeRigidBodies": True,
    }
    data = dict(stage.GetRootLayer().customLayerData)
    data["navmeshSettings"] = values
    stage.GetRootLayer().customLayerData = data
    return values


def attach_moving_human_collider(stage, rig, *, radius=0.4, height=1.7):
    """Kinematic guide capsule follows the moving SkelRoot, not fixed Server.

    This is an explicit scene collision envelope, not calibrated human anatomy.
    No physical mass is authored and no collision metadata enters estimators.
    """
    import math

    if (
        not math.isfinite(radius)
        or not 0.25 <= radius <= 0.5
        or not math.isfinite(height)
        or not 1.5 <= height <= 2
        or 2 * radius >= height
    ):
        raise ValueError("invalid bounded human collision envelope")
    collider = UsdGeom.Capsule.Define(
        stage, str(rig.GetPath()) + "/CRAAnimatedCollision"
    )
    collider.CreateAxisAttr("Z")
    collider.CreateRadiusAttr(radius)
    collider.CreateHeightAttr(height - 2 * radius)
    collider.AddTranslateOp().Set(Gf.Vec3d(0, 0, height / 2))
    collider.CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    body = UsdPhysics.RigidBodyAPI.Apply(collider.GetPrim())
    body.CreateKinematicEnabledAttr(True)
    return collider


def prepare_native_restaurant_human(stage, *, crossing_direction="right-to-left",
                                    crossing_extension_m=0., crossing_y_m=8.9, crossing_layout="legacy", parallel_oncoming=False,
                                    upper_departure=False, straight_service_walk=False):
    import omni.client
    import omni.kit.commands
    from pxr import NavSchema, Sdf, Usd

    result, _, content = omni.client.read_file(HUMAN_MOTION_LIBRARY_URL)
    if (
        result != omni.client.Result.OK
        or hashlib.sha256(bytes(content)).hexdigest() != HUMAN_MOTION_LIBRARY_SHA256
    ):
        raise ValueError(
            "native human motion library unavailable or fingerprint changed"
        )
    server = stage.GetPrimAtPath(SERVER_PRIM_PATH)
    from robot.scene.rear_pass_encounter import validate_crossing_layout, OPEN_ACTOR_START_WORLD
    open_layout = validate_crossing_layout(crossing_layout, direction=crossing_direction,
        extension=crossing_extension_m, crossing_y=crossing_y_m)
    if crossing_direction not in {"right-to-left", "left-to-right"}:
        raise ValueError("unsupported authored crossing direction")
    if crossing_direction == "left-to-right":
        # Explicit session-only actor fixture; never estimator/control data.
        server.GetAttribute("xformOp:translate").Set(Gf.Vec3d(27.8, 8.9, 0.))
    if crossing_extension_m or crossing_y_m != 8.9:
        from robot.scene.restaurant_departing_pedestrian import DepartingPedestrianRoute
        DepartingPedestrianRoute(crossing_direction=crossing_direction,
            crossing_extension_m=crossing_extension_m,crossing_y_m=crossing_y_m)
        start_x = 27.8-crossing_extension_m if crossing_direction=='left-to-right' else 29.8+crossing_extension_m
        server.GetAttribute("xformOp:translate").Set(Gf.Vec3d(start_x,crossing_y_m,0.))
    if open_layout:
        server.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*OPEN_ACTOR_START_WORLD))
    if parallel_oncoming:
        from robot.scene.parallel_walk_encounter import validate_parallel_scope, PARALLEL_ACTOR_START
        validate_parallel_scope(direction=crossing_direction, extension=crossing_extension_m,
            crossing_y=crossing_y_m, layout=crossing_layout)
        server.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*PARALLEL_ACTOR_START))
    if straight_service_walk:
        if not parallel_oncoming or upper_departure:
            raise ValueError('one-way service walk requires its separate parallel setup')
        from robot.scene.restaurant_straight_walk import STRAIGHT_START
        server.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*STRAIGHT_START))
    rigs = [p for p in Usd.PrimRange(server) if p.IsA(UsdSkel.Root)]
    skeletons = [p for p in Usd.PrimRange(server) if p.IsA(UsdSkel.Skeleton)]
    if len(rigs) != 1 or len(skeletons) != 1:
        raise ValueError("native human needs one unambiguous rig")
    rig, skeleton = rigs[0], skeletons[0]
    # The earlier scene's authored static pose must not override native motion.
    UsdSkel.BindingAPI.Apply(skeleton).CreateAnimationSourceRel().SetTargets([])
    old = stage.GetPrimAtPath(SERVER_PRIM_PATH + "/StaticCollision")
    if old.IsValid():
        old.SetActive(False)
    collider = attach_moving_human_collider(stage, rig)
    rig.ApplyAPI(NavSchema.NavMeshExcludeAPI)
    library = stage.DefinePrim(HUMAN_MOTION_LIBRARY_PRIM)
    library.GetReferences().AddReference(HUMAN_MOTION_LIBRARY_URL)
    omni.kit.commands.execute(
        "ApplyBehaviorAgentAPICommand",
        skelroot_prim_paths=[rig.GetPath()],
        motion_library_prim_path=Sdf.Path(HUMAN_MOTION_LIBRARY_PRIM),
        motion_library_skeleton_rig="Human",
    )
    if "BehaviorAgentAPI" not in rig.GetAppliedSchemas():
        raise RuntimeError("native behavior API was not applied")
    volume = NavSchema.NavMeshVolume.Define(
        stage, "/World/CRAServiceActors/HumanNavMeshVolume"
    )
    volume.CreateNavVolumeTypeAttr("Include")
    extent = human_navmesh_extent(open_layout=open_layout, parallel_oncoming=parallel_oncoming,
        upper_departure=upper_departure, straight_service_walk=straight_service_walk)
    volume.CreateExtentAttr([Gf.Vec3f(*corner) for corner in extent])
    settings = configure_human_navmesh(stage)
    return {
        "rig_path": str(rig.GetPath()),
        "collider_path": str(collider.GetPath()),
        "motion_library_url": HUMAN_MOTION_LIBRARY_URL,
        "motion_library_sha256": HUMAN_MOTION_LIBRARY_SHA256,
        "navmesh_settings_metres": settings,
        "actor_navmesh_extent_m": extent,
        "collision_radius_m": 0.4,
        "collision_height_m": 1.7,
        "scope": "native actor motion/render/independent evaluation only; no estimator inputs",
        "authored_crossing_direction": crossing_direction,
        "authored_crossing_layout": crossing_layout,
        "authored_parallel_oncoming": parallel_oncoming,
        "straight_service_walk": straight_service_walk,
    }
