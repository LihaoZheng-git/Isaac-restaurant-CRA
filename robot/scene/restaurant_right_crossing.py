"""Native right-aisle performers with independent physics readback."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdSkel

from experiment.scene.resources import live_assets
from robot.scene.controlled_semantic_obstacle import read_world_bounds
from robot.scene.restaurant_human_motion import (
    HUMAN_MOTION_LIBRARY_SHA256,
    HUMAN_MOTION_LIBRARY_URL,
    configure_human_navmesh,
)


ROOT = "/World/RestaurantLiveV1"
ACTOR_ROOT = ROOT + "/Actors"
CART_PATH = ROOT + "/RightCrossingPlateTrolley"
PARKED_PREP_TROLLEY_PATH = ROOT + "/PinkPrepTrolley"
LIBRARY_PATH = ROOT + "/RightCrossingMotionLibrary"
NAVMESH_VOLUME_PATH = ROOT + "/RightCrossingNavMeshVolume"
ASSETS = live_assets()
BOY_SOURCE = ASSETS / "restaurant_live_v5/characters/rocketbox/Male_Child_01.usdc"
BOY_SHA256 = "a84ed00c370993d8be94987a36887a57683230304dcfc3bb1ed53f6d798a0461"
ATTENDANT_SOURCE = ASSETS / (
    "restaurant_live_v3/characters/official-pack/Assets/Characters/Reallusion/ActorCore/"
    "Uniform_M_0001/Actor/uniform_m_0001/uniform_m_0001.usd"
)
ATTENDANT_SHA256 = "9ec4fa2c538499f9351cdd8ca86d13c11a8ab1929d7af0dd71615c946bc525f0"
CART_SOURCE = ASSETS / "restaurant_live_v1/assets/plate_trolley.usdc"
CART_SHA256 = "ae60c558112943ade4a4d668fbfb34493995d335602355b5582a96ca68a8a580"
CART_OFFSET_M = .69
# The direct second leg stopped at a tight NavMesh bend near (28.34, 6.07).
# Move north of that bend, cross at the same marks, then run behind the bar.
BOY_CROSSING_ROUTE = ((30.30, 3.55, 0.), (29.30, 5.30, 0.), (28.55, 6.60, 0.), (27.25, 7.30, 0.))
BOY_DEPARTURE_ROUTE = (
    (24.00, 7.46, 0.), (20.00, 7.46, 0.), (16.00, 7.46, 0.), (12.00, 7.46, 0.), (8.00, 7.46, 0.),
)
BOY_ROUTE = BOY_CROSSING_ROUTE + BOY_DEPARTURE_ROUTE
BOY_RUN_SPEED_MPS = 1.50
BOY_INTERMEDIATE_HANDOFF_M = .12
TROLLEY_ROUTE = ((27.53, 14.60, 0.), (32.86, 14.60, 0.))
TROLLEY_MAX_CONTINUATIONS = 3
TROLLEY_MIN_CONTINUATION_PROGRESS_M = .10
TROLLEY_HAND_RETRY_SECONDS = 1.


def author_right_crossing_volume(stage, *, include_boy):
    """Stage the tested actor-only navigation volume in the session layer."""
    from pxr import NavSchema

    west = 7.4 if include_boy else 25.5
    extent = [(west, 2.5, -.2), (34.2, 15.7, 2.5)]
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        volume = NavSchema.NavMeshVolume.Define(stage, NAVMESH_VOLUME_PATH)
        volume.CreateNavVolumeTypeAttr("Include")
        volume.CreateExtentAttr([Gf.Vec3f(*corner) for corner in extent])
    return extent


def _verify_run_path(mesh, route):
    """Check the sampled native paths through all staged goals on one baked floor island."""
    maximum_snap = 0.
    maximum_spacing = .04
    snap_limit = .06
    sample_count = 0
    expected_island = None
    for start, end in zip(route, route[1:]):
        count = max(2, math.ceil(math.dist(start, end) / maximum_spacing) + 1)
        sample_count += count
        for point in np.linspace(start, end, count):
            target = tuple(map(float, point))
            closest, island = mesh.query_closest_point(target)
            if closest is None or island < 0 or (expected_island is not None and island != expected_island):
                raise RuntimeError(f"Right crossing boy run curve leaves its floor island at {target}")
            expected_island = island
            snap = math.dist(_xyz(closest), target)
            if snap > snap_limit:
                raise RuntimeError(f"Right crossing boy run curve is off the baked floor at {target}: {snap:.3f}m")
            maximum_snap = max(maximum_snap, snap)
    return dict(method="native move_to waypoint sequence", path=route, island=expected_island,
                path_length_m=sum(math.dist(a, b) for a, b in zip(route, route[1:])),
                sample_count=sample_count, maximum_sample_spacing_m=maximum_spacing,
                maximum_allowed_navmesh_snap_m=snap_limit, maximum_navmesh_snap_m=maximum_snap)


def _verified_asset(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Right crossing asset fingerprint changed: {path}")
    return actual


def _xyz(value):
    return [float(value[index]) for index in range(3)]


def _capsule(stage, rig, radius, height):
    """Use metre dimensions even under the centimetre attendant rig."""
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(rig)
    scales = Gf.Transform(matrix).GetScale()
    scale = float(scales[0])
    if scale <= 0. or not all(math.isclose(s, scale, rel_tol=1e-5) for s in scales):
        raise ValueError("Right crossing character requires uniform positive scaling")
    up = matrix.GetInverse().TransformDir(Gf.Vec3d(0., 0., 1.))
    up.Normalize()
    capsule = UsdGeom.Capsule.Define(stage, str(rig.GetPath()) + "/CRAAnimatedCollision")
    capsule.CreateAxisAttr("Z")
    capsule.CreateRadiusAttr(radius / scale)
    capsule.CreateHeightAttr((height - 2. * radius) / scale)
    capsule.AddTranslateOp().Set(up * (height / (2. * scale)))
    capsule.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0., 0., 1.), up).GetQuat()))
    capsule.CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim()).CreateKinematicEnabledAttr(True)
    return str(capsule.GetPath())


def _author_character(stage, key, route, library, manifest, nav_schema):
    import omni.kit.commands
    from pxr import BehaviorSchema

    boy = key == "boy"
    name, source, expected = (
        ("FamilyChild", BOY_SOURCE, BOY_SHA256) if boy
        else ("Busboy", ATTENDANT_SOURCE, ATTENDANT_SHA256)
    )
    if name in manifest:
        source = Path(manifest[name]["asset_path"])
        expected = manifest[name]["sha256"]
    fingerprint = _verified_asset(source, expected)
    source_stage = Usd.Stage.Open(str(source))
    if not source_stage or UsdGeom.GetStageUpAxis(source_stage) != "Z":
        raise ValueError(f"Right crossing actor requires a Z-up character: {source}")
    path = ACTOR_ROOT + "/" + name
    if stage.GetPrimAtPath(path):
        raise ValueError(f"Right crossing actor already exists: {path}")
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*route[0]))
    units = UsdGeom.Xform.Define(stage, path + "/Units")
    units.AddScaleOp().Set(Gf.Vec3d(UsdGeom.GetStageMetersPerUnit(source_stage)))
    visual = UsdGeom.Xform.Define(stage, path + "/Units/Visual").GetPrim()
    visual.GetReferences().AddReference(source.resolve().as_posix())
    rig, = (p for p in Usd.PrimRange(visual) if p.IsA(UsdSkel.Root))
    skeleton, = (p for p in Usd.PrimRange(visual) if p.IsA(UsdSkel.Skeleton))
    UsdSkel.BindingAPI.Apply(skeleton).CreateAnimationSourceRel().SetTargets([])
    rig.ApplyAPI(nav_schema.NavMeshExcludeAPI)
    radius, height = (.25, 1.45) if boy else (.32, 1.70)
    collider = _capsule(stage, rig, radius, height)
    if boy:
        # The exported Rocketbox rig already includes the Human retarget map.
        behavior = BehaviorSchema.BehaviorAgentAPI.Apply(rig)
        behavior.CreateBehaviorMotionLibraryRel().SetTargets([library.GetPath()])
    else:
        omni.kit.commands.execute(
            "ApplyBehaviorAgentAPICommand",
            skelroot_prim_paths=[rig.GetPath()],
            motion_library_prim_path=library.GetPath(),
            motion_library_skeleton_rig="Human",
        )
    if "BehaviorAgentAPI" not in rig.GetAppliedSchemas():
        raise RuntimeError(f"Native right crossing behavior API absent: {name}")
    tags = list(skeleton.GetAttribute("controlRig:retargetTags").Get() or [])
    joints = {tag: tags.index(tag) for tag in ("LeftFoot", "RightFoot", "LeftHand", "RightHand")
              if tag in tags}
    return dict(root_path=path, rig_path=str(rig.GetPath()), collider_path=collider,
                radius_m=radius, height_m=height, route=route, joints=joints,
                asset_path=str(source.resolve()), asset_sha256=fingerprint,
                speed_mps=BOY_RUN_SPEED_MPS if boy else .55, robot_trigger_y_m=5.35 if boy else 12.2)


class RightCrossingEpisode:
    """Drive native actor tasks only. Records and scripted poses are evaluation data."""

    def __init__(self, stage, case, output, cast_manifest=None):
        import omni.client
        from pxr import NavSchema

        cases = {"boy": ("boy",), "trolley": ("trolley",),
                 "combined": ("boy", "trolley"), "delivery": ("boy", "trolley")}
        if case not in cases:
            raise ValueError(f"Unknown right crossing case: {case}")
        self.stage = stage
        self.output = Path(output)
        self.records_by_actor = {key: [] for key in cases[case]}
        self.events = []
        self.actors = {}
        self.cart = None
        entries = {}
        if cast_manifest is not None:
            manifest = Path(cast_manifest).resolve()
            entries = json.loads(manifest.read_text(encoding="utf-8"))["actors"]
            entries = {
                key: {**entry, "asset_path": str(manifest.parent / entry["asset_path"])}
                for key, entry in entries.items() if key in ("FamilyChild", "Busboy") and "asset_path" in entry
            }
        status, _, content = omni.client.read_file(HUMAN_MOTION_LIBRARY_URL)
        if (status != omni.client.Result.OK
                or hashlib.sha256(bytes(content)).hexdigest() != HUMAN_MOTION_LIBRARY_SHA256):
            raise RuntimeError("Verified native locomotion library unavailable")
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            parked_trolley = stage.GetPrimAtPath(PARKED_PREP_TROLLEY_PATH)
            parked_trolley_was_active = bool(parked_trolley and parked_trolley.IsActive())
            if parked_trolley_was_active and not parked_trolley.SetActive(False):
                raise RuntimeError("Could not deactivate the parked restaurant prep trolley")
            library = stage.DefinePrim(LIBRARY_PATH)
            library.GetReferences().AddReference(HUMAN_MOTION_LIBRARY_URL)
            extent = author_right_crossing_volume(stage, include_boy="boy" in self.records_by_actor)
            actor_setup = {}
            for key in self.records_by_actor:
                route = BOY_ROUTE if key == "boy" else TROLLEY_ROUTE
                actor_setup[key] = _author_character(stage, key, route, library, entries, NavSchema)
                if key == "boy":
                    actor_setup[key]["departure_marks"] = BOY_DEPARTURE_ROUTE
                    actor_setup[key]["intermediate_handoff_m"] = BOY_INTERMEDIATE_HANDOFF_M
                else:
                    actor_setup[key].update(
                        maximum_early_success_continuations=TROLLEY_MAX_CONTINUATIONS,
                        minimum_continuation_progress_m=TROLLEY_MIN_CONTINUATION_PROGRESS_M,
                        hand_retry_interval_seconds=TROLLEY_HAND_RETRY_SECONDS,
                    )
            cart_setup = self._author_cart() if "trolley" in actor_setup else None
            if cart_setup is not None:
                self.cart.GetPrim().ApplyAPI(NavSchema.NavMeshExcludeAPI)
        root_layer = stage.GetRootLayer()
        old_metadata = dict(root_layer.customLayerData)
        try:
            self.navmesh_settings = configure_human_navmesh(stage)
        finally:
            root_layer.customLayerData = old_metadata
        self.setup = dict(case=case, actors=actor_setup, trolley=cart_setup, navmesh_extent_m=extent,
                          parked_prep_trolley=dict(prim_path=PARKED_PREP_TROLLEY_PATH,
                                                   active_before=parked_trolley_was_active, active_after=False),
                          navmesh_settings_metres=self.navmesh_settings,
                          motion_library_sha256=HUMAN_MOTION_LIBRARY_SHA256,
                          scope="Native actor and collider readbacks for evaluation. No authored pose enters navigation.")
        self.contact_targets = {}
        self.actor_radii_m = {key: spec["radius_m"] for key, spec in actor_setup.items()}
        if "boy" in actor_setup:
            self.contact_targets["boy"] = actor_setup["boy"]["collider_path"]
        if cart_setup is not None:
            self.contact_targets.update(trolley=cart_setup["collider_path"],
                                        server=actor_setup["trolley"]["collider_path"])

    def _author_cart(self):
        stage = self.stage
        previous = stage.GetPrimAtPath(ROOT + "/PlateTrolley")
        if previous and previous.IsActive():
            previous.SetActive(False)
        self.cart = UsdGeom.Xform.Define(stage, CART_PATH)
        self.cart_translate = self.cart.AddTranslateOp()
        x, y, z = TROLLEY_ROUTE[0]
        self.cart_translate.Set(Gf.Vec3d(x + CART_OFFSET_M, y, z))
        visual = UsdGeom.Xform.Define(stage, CART_PATH + "/Visual")
        visual.GetPrim().GetReferences().AddReference(CART_SOURCE.as_posix())
        visual.AddRotateZOp().Set(180.)
        bounds = read_world_bounds(stage, str(visual.GetPath()))
        self.cart_low = tuple(bounds.minimum[i] - (x + CART_OFFSET_M, y, z)[i] for i in range(3))
        self.cart_high = tuple(bounds.maximum[i] - (x + CART_OFFSET_M, y, z)[i] for i in range(3))
        # Give physics its own world-space kinematic body. Updating a child's
        # USD parent xform alone need not update the running PhysX pose.
        collider = UsdGeom.Cube.Define(stage, ROOT + "/RightCrossingTrolleyCollision")
        collider.SetResetXformStack(True)
        collider.CreatePurposeAttr("guide")
        collider.CreateSizeAttr(1.)
        collider_low = (*self.cart_low[:2], .02)
        self.cart_collision_center = tuple((a + b) / 2 for a, b in zip(collider_low, self.cart_high))
        collider.AddTranslateOp().Set(Gf.Vec3d(
            x + CART_OFFSET_M + self.cart_collision_center[0],
            y + self.cart_collision_center[1], self.cart_collision_center[2],
        ))
        collider.AddScaleOp().Set(Gf.Vec3d(*(b - a for a, b in zip(collider_low, self.cart_high))))
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.RigidBodyAPI.Apply(collider.GetPrim()).CreateKinematicEnabledAttr(True)
        self.cart_wheels = [UsdGeom.Xformable(p).AddRotateZOp(opSuffix="liveSpin")
                            for p in Usd.PrimRange(visual.GetPrim())
                            if p.IsA(UsdGeom.Xform) and p.GetName().lower().startswith("wheel_")]
        self.push_targets = {}
        for hand, lateral in (("LeftHand", .115), ("RightHand", -.115)):
            target = UsdGeom.Xform.Define(stage, CART_PATH + "/" + hand + "Push")
            target.AddTranslateOp().Set(Gf.Vec3d(-.4524, lateral, .8577))
            self.push_targets[hand] = str(target.GetPath())
        return dict(prim_path=CART_PATH, collider_path=str(collider.GetPath()),
                    asset_path=str(CART_SOURCE), asset_sha256=_verified_asset(CART_SOURCE, CART_SHA256),
                    bounds_size_m=list(bounds.size), driver_offset_m=CART_OFFSET_M,
                    animated_wheel_groups=len(self.cart_wheels))

    def bake(self):
        import omni.anim.navigation.core as nav

        root = self.stage.GetRootLayer()
        metadata = dict(root.customLayerData)
        try:
            root.customLayerData = {**metadata, "navmeshSettings": self.navmesh_settings}
            api = nav.acquire_interface()
            if not api.start_navmesh_baking_and_wait():
                raise RuntimeError("Native right crossing NavMesh bake failed")
        finally:
            root.customLayerData = metadata
        mesh = api.get_navmesh()
        if not mesh:
            raise RuntimeError("Native right crossing NavMesh unavailable")
        for key, spec in self.setup["actors"].items():
            route = spec["route"]
            checks = []
            for a, b in zip(route, route[1:]):
                start, island = mesh.query_closest_point(a)
                goal, goal_island = mesh.query_closest_point(b)
                if (island < 0 or island != goal_island or math.dist(_xyz(start), a) > .20
                        or math.dist(_xyz(goal), b) > .20):
                    raise RuntimeError(f"Right crossing {key} endpoint outside its NavMesh: {a}, {b}")
                path = mesh.query_shortest_path(start, goal)
                points = [_xyz(p) for p in path.get_points()] if path else []
                length = sum(math.dist(p, q) for p, q in zip(points, points[1:]))
                if len(points) < 2 or length > math.dist(a, b) + .75:
                    raise RuntimeError(f"Right crossing {key} NavMesh detours from the staged route")
                checks.append(dict(start=_xyz(start), goal=_xyz(goal), island=island, points=points,
                                   length_m=length))
            spec["navmesh_legs"] = checks
            if key == "boy":
                baked_route = tuple(point for leg in checks for point in leg["points"])
                spec["run_path"] = _verify_run_path(mesh, baked_route)
        if "boy" in self.records_by_actor:
            boy = self.setup["actors"]["boy"]
            receipt = dict(run_path=boy["run_path"], departure_marks=boy["departure_marks"],
                           speed_mps=boy["speed_mps"], robot_trigger_y_m=boy["robot_trigger_y_m"],
                           intermediate_handoff_m=boy["intermediate_handoff_m"],
                           navmesh_extent_m=self.setup["navmesh_extent_m"],
                           navmesh_settings_metres=self.navmesh_settings)
            (self.output / "right-crossing-run-path.json").write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )

    def initialize(self):
        import carb
        import omni.anim.behavior.core as bh
        import warp as wp
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager

        interface = bh.acquire_interface()
        for key, spec in self.setup["actors"].items():
            agent = interface.get_agent(spec["rig_path"])
            if agent is None:
                raise RuntimeError(f"Native right crossing agent unavailable: {key}")
            agent.set_auto_avoidance_enabled(False)
            agent.set_auto_impact_enabled(False)
            a, b = spec["route"][:2]
            facing = Gf.Vec3d(*(j - i for i, j in zip(a, b))).GetNormalized()
            agent.idle(facing=carb.Float3(*facing))
            body = RigidPrim(spec["collider_path"])
            if not body.is_physics_tensor_entity_valid():
                raise RuntimeError(f"Right crossing actor collider missing from physics: {key}")
            self.actors[key] = dict(agent=agent, body=body, phase="idle_before", leg=0,
                                    task=None, leg_retry=False)
        if self.cart is not None:
            self.cart_body = RigidPrim(self.setup["trolley"]["collider_path"])
            if not self.cart_body.is_physics_tensor_entity_valid():
                raise RuntimeError("Right crossing trolley collider missing from physics")
            self.cart_simulation_view = SimulationManager.get_physics_simulation_view()
            if self.cart_simulation_view is None:
                raise RuntimeError("Right crossing trolley requires the running PhysX simulation view")
            self.cart_rigid_view = self.cart_simulation_view.create_rigid_body_view(
                self.setup["trolley"]["collider_path"]
            )
            if self.cart_rigid_view.count != 1:
                raise RuntimeError("Right crossing trolley needs one kinematic physics body")
            self.cart_indices = wp.array([0], dtype=wp.uint32, device=self.cart_simulation_view.device)

    @property
    def positions(self):
        """Evaluation-only native positions, keyed by performer."""
        return {key: _xyz(actor["agent"].get_world_translation()) for key, actor in self.actors.items()}

    def _move(self, key, actor, now, xyz):
        import omni.anim.behavior.core as bh

        spec = self.setup["actors"][key]
        goal = spec["route"][actor["leg"] + 1]
        final = actor["leg"] + 2 == len(spec["route"])
        auto_brake = key != "boy" or final
        actor["agent"].set_speed(spec["speed_mps"])
        actor["task"] = actor["agent"].move_to(goal, auto_brake=auto_brake)
        if actor["task"] == bh.BEHAVIOR_TASK_ID_INVALID:
            raise RuntimeError(f"Native right crossing task was rejected: {key}")
        actor["phase"] = "move"
        if key == "trolley":
            actor["move_start_xyz"] = xyz
        self.events.append(dict(time=now, actor=key, event="move", leg=actor["leg"],
                                goal=goal, auto_brake=auto_brake, speed_mps=spec["speed_mps"],
                                method="move_to", task_id=actor["task"], actor_xyz=xyz))

    def _advance(self, key, actor, now, xyz):
        actor["leg"] += 1
        actor["leg_retry"] = False
        if actor["leg"] + 1 < len(self.setup["actors"][key]["route"]):
            self._move(key, actor, now, xyz)
        else:
            actor["agent"].idle()
            actor["phase"] = "away_endpoint"
            self.events.append(dict(time=now, actor=key, event="away_endpoint", actor_xyz=xyz))

    def _continue_trolley(self, actor, spec, now, xyz, goal, gap, status):
        """Reissue the same goal only after a premature success with measured progress."""
        start = actor["move_start_xyz"]
        progress = min(xyz[0] - start[0], math.dist(start, goal) - gap)
        continuations = actor.get("move_continuations", 0)
        if (progress <= spec["minimum_continuation_progress_m"]
                or continuations >= spec["maximum_early_success_continuations"]):
            return False
        actor["move_continuations"] = continuations + 1
        self.events.append(dict(time=now, actor="trolley", event="locomotion_early_success_continuation",
                                leg=actor["leg"], goal=goal, actor_xyz=xyz, task_id=actor["task"],
                                previous_start_xyz=start, measured_progress_m=progress,
                                goal_error_m=gap, status=status, continuation=continuations + 1))
        self._move("trolley", actor, now, xyz)
        return True

    def _cart_row(self, actor, xyz, now):
        import omni.anim.behavior.core as bh
        import warp as wp

        start = TROLLEY_ROUTE[0]
        x = max(float(self.cart_translate.Get()[0]), xyz[0] + CART_OFFSET_M)
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self.cart_translate.Set(Gf.Vec3d(x, start[1], 0.))
            angle = -math.degrees((x - start[0] - CART_OFFSET_M) / .0579532)
            for wheel in self.cart_wheels:
                wheel.Set(angle)
        collider_center = [x + self.cart_collision_center[0],
                           start[1] + self.cart_collision_center[1], self.cart_collision_center[2]]
        target = wp.array([[*collider_center, 0., 0., 0., 1.]], dtype=wp.float32,
                          device=self.cart_simulation_view.device)
        self.cart_rigid_view.set_kinematic_targets(target, self.cart_indices)
        push_tasks = actor.setdefault("push_tasks", {})
        push_readbacks = {}
        agent = actor["agent"]
        for hand, usage in (("LeftHand", bh.BehaviorHandUsage.LEFT_HAND),
                            ("RightHand", bh.BehaviorHandUsage.RIGHT_HAND)):
            task, issued = push_tasks.get(hand, (bh.BEHAVIOR_TASK_ID_INVALID, -math.inf))
            status = (None if task == bh.BEHAVIOR_TASK_ID_INVALID else agent.get_task_status(task))
            retry_interval = self.setup["actors"]["trolley"]["hand_retry_interval_seconds"]
            if (actor["phase"] == "move" and status != bh.BehaviorTaskStatus.RUNNING
                    and now - issued >= retry_interval):
                previous_task, previous_status = task, status
                task = agent.reach_hand(usage, self.push_targets[hand])
                push_tasks[hand] = task, now
                status = (None if task == bh.BEHAVIOR_TASK_ID_INVALID else agent.get_task_status(task))
                self.events.append(dict(time=now, actor="trolley", event="push_reach", hand=hand,
                                        task_id=task, previous_task_id=previous_task,
                                        previous_status=None if previous_status is None else str(previous_status),
                                        locomotion_task_id=actor["task"]))
            push_readbacks[hand] = dict(task_id=task, status=None if status is None else str(status))
        return dict(trolley_world_xy=[x, start[1]], trolley_bounds_world_xy=[
            [x + self.cart_low[0], start[1] + self.cart_low[1]],
            [x + self.cart_high[0], start[1] + self.cart_high[1]],
        ], trolley_collider_path=self.setup["trolley"]["collider_path"],
            trolley_collider_target_world_xyz=collider_center, push_hand_tasks=push_readbacks)

    def update(self, now, robot_xy, *, navigation=False):
        import carb
        import omni.anim.behavior.core as bh
        from isaacsim.core.experimental.utils.backend import use_backend

        for key, xyz in self.positions.items():
            actor, spec = self.actors[key], self.setup["actors"][key]
            failure = None
            task_status = None
            status_task = None
            if navigation and actor["phase"] == "idle_before" and robot_xy[1] >= spec["robot_trigger_y_m"]:
                self._move(key, actor, now, xyz)
            elif actor["phase"] == "move":
                status_task = actor["task"]
                status = actor["agent"].get_task_status(status_task)
                task_status = str(status)
                goal = spec["route"][actor["leg"] + 1]
                final = actor["leg"] + 2 == len(spec["route"])
                gap = math.dist(xyz, goal)
                if status == bh.BehaviorTaskStatus.FAILED:
                    self.events.append(dict(time=now, actor=key, event="locomotion_failed",
                                            leg=actor["leg"], goal=goal, actor_xyz=xyz, status=task_status))
                    if key == "boy" and navigation and not actor["leg_retry"]:
                        actor["leg_retry"] = True
                        self.events.append(dict(time=now, actor=key, event="locomotion_retry",
                                                leg=actor["leg"], goal=goal, actor_xyz=xyz))
                        self._move(key, actor, now, xyz)
                    else:
                        failure = f"Native right crossing locomotion failed: {key}"
                elif key == "boy" and not final and gap <= spec["intermediate_handoff_m"]:
                    self.events.append(dict(time=now, actor=key, event="run_through_mark",
                                            leg=actor["leg"], goal=goal, actor_xyz=xyz,
                                            goal_error_m=gap, status=task_status))
                    self._advance(key, actor, now, xyz)
                elif status == bh.BehaviorTaskStatus.SUCCEEDED:
                    if gap <= .24:
                        self._advance(key, actor, now, xyz)
                    elif key == "trolley" and navigation and self._continue_trolley(
                            actor, spec, now, xyz, goal, gap, task_status):
                        pass
                    else:
                        self.events.append(dict(time=now, actor=key, event="locomotion_goal_mismatch",
                                                leg=actor["leg"], goal=goal, actor_xyz=xyz,
                                                goal_error_m=gap, status=task_status, task_id=status_task))
                        failure = f"Native right crossing actor ended away from its goal: {key}"
            if failure:
                actor["phase"] = "task_failed"
                actor["agent"].idle()
            with use_backend("tensor", raise_on_fallback=True):
                collision_xyz = _xyz(actor["body"].get_world_poses()[0].numpy()[0])
                cart_collision_xyz = (_xyz(self.cart_body.get_world_poses()[0].numpy()[0])
                                      if key == "trolley" else None)
            joints = {}
            for name, index in spec["joints"].items():
                position, orientation = carb.Float3(), carb.Float4()
                if actor["agent"].get_joint_world_transform(index, position, orientation):
                    joints[name] = _xyz(position)
            row = dict(time=float(now), phase=actor["phase"], actor_xyz=xyz, joints_world_xyz=joints,
                       task_status=task_status, task_status_task_id=status_task, task_leg=actor["leg"],
                       collider_path=spec["collider_path"], collider_world_xyz=collision_xyz,
                       collider_alignment_error_m=math.dist(collision_xyz,
                                                           [*xyz[:2], xyz[2] + spec["height_m"] / 2]))
            if key == "trolley":
                row.update(self._cart_row(actor, xyz, now), trolley_collider_world_xyz=cart_collision_xyz)
                row["trolley_collider_alignment_error_m"] = math.dist(
                    cart_collision_xyz, row["trolley_collider_target_world_xyz"]
                )
            self.records_by_actor[key].append(row)
            if failure:
                raise RuntimeError(failure)

    def abort(self):
        for actor in self.actors.values():
            if actor["phase"] == "move":
                actor["agent"].cancel_task(actor["task"])
                actor["agent"].idle()

    def close(self):
        path = self.output / "right-crossing-actors.json"
        path.write_text(json.dumps(dict(setup=self.setup, events=self.events,
                                        records_by_actor=self.records_by_actor), indent=2) + "\n",
                        encoding="utf-8")
