"""Optional service animation around the physical Carter and its existing tray."""

import hashlib
import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

from robot.scene.controlled_semantic_obstacle import read_world_bounds
from robot.scene.isaac_restaurant_server import (
    ISAAC_HUMAN_SHA256, ISAAC_HUMAN_URL, author_relaxed_static_pose, validate_isaac_human_bytes,
)
from robot.scene.restaurant_demo_build import PAYLOAD_PATH
from robot.scene.restaurant_live_bar_handoff import BarHandoff


MEAL_PATH = "/World/RestaurantLiveV1/PhysicalHandoffMeal"
BAR_DOCK = (25.80, 6.25)
BAR_SERVER = (25.31, 6.10, 0.)
BAR_PERFORMER_PATH = "/World/RestaurantLiveV1/StaticActors/BarServer"
HANDOFF_DURATION_S = 4.6
MAXIMUM_DROP_M = .25
MAXIMUM_WRIST_ERROR_M = .03
MAXIMUM_PARKED_TRAY_DRIFT_M = .01
_GRAVITY_M_S2 = 9.81


class _SurfaceSpot(NamedTuple):
    path: str
    xyz: tuple[float, float, float]


_PICKUP = _SurfaceSpot(
    "/World/Environment/Restaurant/rh_Restaurant/Geometry/G_Arch/G_Bar/bar_ctop_low", (24.80, 6.10, .90))
_DELIVERY = _SurfaceSpot(
    "/World/Environment/Restaurant/assets_restaurant/Assets_Banquettes_Left02/Geo_TableSet_09/rh_table",
    (4.23, 16.81, .7466))


def _visibility(prim, visible):
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(
        UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
    )


def _bar_staff(stage, manifest_path, *, performer_path=BAR_PERFORMER_PATH):
    import omni.client

    entry = json.loads(manifest_path.read_text(encoding="utf-8"))["actors"]["DinerA"]
    if (entry.get("asset_url"), entry.get("sha256")) != (ISAAC_HUMAN_URL, ISAAC_HUMAN_SHA256):
        raise ValueError("The physical handoff requires the rehearsed official bar-server rig")
    status, _, data = omni.client.read_file(ISAAC_HUMAN_URL)
    if status != omni.client.Result.OK:
        raise RuntimeError(f"Could not verify the official bar-server asset: {status}")
    validate_isaac_human_bytes(bytes(data))
    actor = UsdGeom.Xform.Define(stage, performer_path)
    grounding = UsdGeom.Xform.Define(stage, performer_path + "/Grounding")
    visual = UsdGeom.Xform.Define(stage, performer_path + "/Grounding/Visual")
    visual.GetPrim().GetReferences().AddReference(ISAAC_HUMAN_URL)
    low, high = author_relaxed_static_pose(stage, visual.GetPrim())
    grounding.AddTranslateOp().Set(Gf.Vec3d(0., 0., -float(low[2])))
    actor.AddTranslateOp().Set(Gf.Vec3d(*BAR_SERVER))
    actor.AddRotateZOp().Set(-90.)
    for part, tint in (("shirt", (.72, .70, .64)), ("vest", (.035, .041, .046))):
        path = f"{visual.GetPath()}/female_adult_business_02/Looks/opaque__fabric__{part}/Shader"
        color = UsdShade.Shader.Get(stage, path).GetInput("diffuse_tint")
        if not color:
            raise RuntimeError(f"Rehearsed bar-server clothing shader missing: {path}")
        color.Set(Gf.Vec3f(*tint))
    return dict(root_path=performer_path, source=ISAAC_HUMAN_URL, sha256=ISAAC_HUMAN_SHA256,
                position=list(BAR_SERVER), facing=[-1., 0., 0.], yaw_degrees=-90.,
                posed_height_m=float(high[2] - low[2]))


def _surface_support(stage, spot, samples):
    triangles = []
    cache = UsdGeom.XformCache()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(spot.path)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        matrix = cache.GetLocalToWorldTransform(prim)
        points = np.asarray([matrix.Transform(point) for point in mesh.GetPointsAttr().Get()])
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        offset = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = points[indices[offset:offset + count]]
            offset += count
            if count >= 3 and np.all(np.abs(face[:, 2] - spot.xyz[2]) < 2.e-5):
                triangles.extend((face[0, :2], face[i, :2], face[i + 1, :2]) for i in range(1, count - 1))
    if not triangles:
        return False
    p0, p1, p2 = np.asarray(triangles).transpose(1, 0, 2)
    a, b = p1 - p0, p2 - p0
    det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    valid = np.abs(det) > 1.e-12
    denominator = np.where(valid, det, 1.)
    for point in samples:
        c = np.asarray(point) - p0
        u = (c[:, 0] * b[:, 1] - c[:, 1] * b[:, 0]) / denominator
        v = (a[:, 0] * c[:, 1] - a[:, 1] * c[:, 0]) / denominator
        if not np.any(valid & (u >= -1.e-5) & (v >= -1.e-5) & (u + v <= 1. + 1.e-5)):
            return False
    return True


class _CarterMealPresentation:
    """Share the mounted meal and verified tabletop between service sequences."""

    def __init__(self, stage, *, meal_path=MEAL_PATH):
        self.stage = stage
        self.meal_path = meal_path
        self.tray = UsdGeom.Cube.Get(stage, PAYLOAD_PATH + "/Tray")
        self.payload = tuple(stage.GetPrimAtPath(PAYLOAD_PATH + "/" + name)
                             for name in ("plate_dinner", "bread"))
        if not self.tray or not all(prim and prim.IsActive() for prim in self.payload):
            raise ValueError("The physical handoff requires Carter's existing serving tray and meal")
        if stage.GetPrimAtPath(self.meal_path):
            raise ValueError(f"The physical handoff meal already exists: {self.meal_path}")
        self.delivery = _DELIVERY
        self.ready = self.delivered = False
        self.records = []
        self.events = []

    def _tray_pose(self):
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(self.tray.GetPrim())
        point = transform.Transform(Gf.Vec3d(0., 0., self.tray.GetSizeAttr().Get() / 2.))
        forward = transform.TransformDir(Gf.Vec3d(1., 0., 0.))
        return point, math.atan2(forward[1], forward[0])

    def _author_meal_visual(self):
        payload_root = self.stage.GetPrimAtPath(PAYLOAD_PATH)
        world_from_payload = UsdGeom.XformCache().GetLocalToWorldTransform(payload_root)
        anchor = world_from_payload.GetInverse().Transform(self.tray_top)
        self.meal = UsdGeom.Xform.Define(self.stage, self.meal_path)
        self.meal.SetResetXformStack(True)
        self.meal_op = self.meal.AddTransformOp()
        self.meal_op.Set(Gf.Matrix4d(1.))
        cloned = UsdGeom.Xform.Define(self.stage, self.meal_path + "/Payload")
        cloned.AddTranslateOp().Set(-anchor)
        for prim in self.payload:
            path = cloned.GetPath().AppendChild(prim.GetName())
            visual = self.stage.DefinePrim(path, "Xform")
            visual.GetReferences().AddInternalReference(prim.GetPath())
            _visibility(visual, True)
            for child in Usd.PrimRange(visual):
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr().Set(False)

    def _set_meal_pose(self, xyz):
        matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0., 0., 1.), math.degrees(self.meal_yaw)))
        matrix.SetTranslateOnly(Gf.Vec3d(*xyz))
        self.meal_op.Set(matrix)

    def _support(self, location):
        self._set_meal_pose(location.xyz)
        bounds = read_world_bounds(self.stage, self.meal_path)
        samples = [(x, y) for x in (bounds.minimum[0], bounds.center[0], bounds.maximum[0])
                   for y in (bounds.minimum[1], bounds.center[1], bounds.maximum[1])]
        if abs(bounds.minimum[2] - location.xyz[2]) > .004 or not _surface_support(self.stage, location, samples):
            raise ValueError(f"Meal footprint lacks support at {location.path}")

    def finish_delivery(self):
        """Show the plate on the verified left tabletop after the caller's stop."""
        if not self.ready or self.delivered:
            raise RuntimeError("Delivery requires one completed loading sequence")
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self._set_meal_pose(self.delivery.xyz)
            for prim in self.payload:
                _visibility(prim, False)
            _visibility(self.meal.GetPrim(), True)
        self.delivered = True
        event = dict(event="meal_placed_on_left_table", world_xyz=list(self.delivery.xyz))
        self.events.append(event)
        return event


class LiveCarterPreloadedDelivery(_CarterMealPresentation):
    """Start with the existing mounted meal and reveal it at the verified table."""

    def __init__(self, stage, *, meal_path=MEAL_PATH):
        super().__init__(stage, meal_path=meal_path)
        if any(UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.inherited
               for prim in (self.tray.GetPrim(), *self.payload)):
            raise ValueError("Preloaded delivery requires the mounted meal and tray to be visible")
        self.tray_top, mounted_yaw = self._tray_pose()
        # Keep the rehearsed tabletop orientation independent of the start pose.
        self.meal_yaw = -math.pi / 2.
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self._author_meal_visual()
            self._support(self.delivery)
            _visibility(self.meal.GetPrim(), False)
        self.ready = True
        tray_xyz = list(self.tray_top)
        self.events.append(dict(event="meal_on_physical_tray", elapsed_sim_s=0.,
                                tray_world_xyz=tray_xyz, basis="preloaded_at_trial_start"))
        self.records.append(dict(elapsed_sim_s=0., phase="preloaded", ready=True, tray_world_xyz=tray_xyz))
        self.metadata = dict(
            measured_tray_top_world_xyz=tray_xyz, measured_tray_yaw_radians=mounted_yaw,
            mounted_meal_path=PAYLOAD_PATH, table_meal_path=self.meal_path, delivery_surface=self.delivery.path,
            delivery_world_xyz=list(self.delivery.xyz), transfer_mode="preloaded_then_scripted_table_placement",
            scope="The existing tray and meal start visible. Table placement is presentation only.",
        )


class LiveCarterBarHandoff(_CarterMealPresentation):
    """Animate the existing visual meal before wheels and after the left stop."""

    def __init__(self, stage, cast_manifest, *, meal_path=MEAL_PATH,
                 performer_path=BAR_PERFORMER_PATH):
        super().__init__(stage, meal_path=meal_path)
        tray, self.tray_yaw = self._tray_pose()
        if math.dist(tray[:2], (BAR_DOCK[0], BAR_DOCK[1] + .04)) > .06:
            raise ValueError("Carter must be settled at the south-facing bar dock")
        if abs(math.remainder(self.tray_yaw + math.pi / 2., math.tau)) > math.radians(5.):
            raise ValueError("Carter must be south-facing for forward departure")
        self.tray_top = tray
        self.meal_yaw = self.tray_yaw
        self.pickup = _PICKUP
        self._loaded = self._performer_cleared = False
        self.elapsed = -math.inf
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            manifest_path = Path(cast_manifest).resolve()
            self.server = _bar_staff(stage, manifest_path, performer_path=performer_path)
            self.handoff = BarHandoff(
                stage, self.server, self.pickup.xyz, (*tray[:2], .93),
                transfer_via=(25.31, 5.65), lift_height=.18, max_lean_degrees=55.,
                plate_radius=.09, allow_crouch=True,
            )
            self.release_z, reach = self._select_release_height()
            self.drop_m = self.release_z - tray[2]
            self.release_s = .88 * HANDOFF_DURATION_S
            self.drop_duration_s = math.sqrt(2. * self.drop_m / _GRAVITY_M_S2)
            self._author_meal_visual()
            self._support(self.pickup)
            self._support(self.delivery)
            self._set_meal_pose(self.pickup.xyz)
            hidden_dressing = []
            for name in ("CounterPlateA", "CounterPlateB"):
                prim = stage.GetPrimAtPath("/World/RestaurantLiveV1/Dressing/" + name)
                if prim and prim.IsActive():
                    _visibility(prim, False)
                    hidden_dressing.append(str(prim.GetPath()))
            for prim in self.payload:
                _visibility(prim, False)
            self.handoff.update(0.)
        self.metadata = dict(
            actor=self.server, measured_tray_top_world_xyz=list(tray), mounted_meal_path=PAYLOAD_PATH, table_meal_path=self.meal_path,
            cast_manifest=str(manifest_path), cast_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            selected_release_z_m=self.release_z, release_drop_m=self.drop_m,
            release_drop_duration_s=self.drop_duration_s, maximum_preflight_wrist_error_m=reach["worst_error_m"],
            selected_crouch_m=self.handoff.crouch_depth_m, reach_preflight=reach,
            pickup_surface=self.pickup.path, delivery_surface=self.delivery.path, hidden_dressing=hidden_dressing,
            handoff_duration_s=HANDOFF_DURATION_S,
            transfer_mode="choreographed_short_drop" if self.drop_m > 1.e-6 else "hand_lower_to_tray",
            scope="Visible handoff and dish placement. Physical Carter and the existing tray are unchanged.",
        )

    def _select_release_height(self):
        # The loaded physical tray is far below the rehearsal tray. Select the
        # lowest height this exact rig can reach through the complete transfer.
        floor = float(self.tray_top[2])
        limit = min(.93, floor + MAXIMUM_DROP_M)
        candidates = [floor + .05 * i for i in range(6) if floor + .05 * i < limit]
        attempts = []
        for depth in (0., .12, .18, .24, .30):
            self.handoff.crouch_depth_m = depth
            for z in (*candidates, limit):
                self.handoff.tray = Gf.Vec3d(*self.tray_top[:2], z)
                phase_errors = {}
                receipt = dict(tray_z_m=floor, release_z_m=z, crouch_depth_m=depth,
                               maximum_wrist_error_by_phase_m=phase_errors)
                try:
                    for progress in (.33, .38, .43, .48, .53, .59, .65, .71, .75, .78, .82, .86, .879):
                        pose = self.handoff.update(progress)
                        phase = pose["phase"]
                        worst = max(hand["error_m"] for hand in pose["wrists"].values())
                        phase_errors[phase] = max(phase_errors.get(phase, 0.), worst)
                except ValueError as error:
                    receipt["pose_error"] = str(error)
                    attempts.append(receipt)
                    break  # Planted-foot geometry is independent of release height.
                receipt["worst_error_m"] = max(phase_errors.values())
                attempts.append(receipt)
                if receipt["worst_error_m"] <= MAXIMUM_WRIST_ERROR_M:
                    return z, receipt
        raise ValueError("No verified visible bar release within 25 cm of the physical tray. "
                         "Per-phase reach attempts: " + json.dumps(attempts, separators=(",", ":")))

    def tick(self, elapsed_sim_s):
        """Step the visual sequence before each zero-command physics step."""
        elapsed = float(elapsed_sim_s)
        if not math.isfinite(elapsed) or elapsed < max(0., self.elapsed):
            raise ValueError("Handoff clock must be finite, nonnegative and nondecreasing")
        if self.ready:
            return self.records[-1]
        self.elapsed = elapsed
        current_tray, _ = self._tray_pose()
        drift = (current_tray - self.tray_top).GetLength()
        if drift > MAXIMUM_PARKED_TRAY_DRIFT_M:
            raise RuntimeError("Carter moved before the bar handoff finished")
        progress = min(1., elapsed / HANDOFF_DURATION_S)
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            receipt = self.handoff.update(progress)
            phase = receipt["phase"]
            if .33 <= progress < .88 and any(
                hand["error_m"] > MAXIMUM_WRIST_ERROR_M for hand in receipt["wrists"].values()
            ):
                raise RuntimeError("The server lost its verified grip before releasing the meal")
            if progress < .88:
                self._set_meal_pose(receipt["dish_xyz"])
            elif elapsed < self.release_s + self.drop_duration_s:
                drop_time = elapsed - self.release_s
                z = self.release_z - .5 * _GRAVITY_M_S2 * drop_time**2
                self._set_meal_pose((*current_tray[:2], z))
                phase = "short_visible_release"
            elif not self._loaded:
                _visibility(self.meal.GetPrim(), False)
                for prim in self.payload:
                    _visibility(prim, True)
                self._loaded = True
                self.events.append(dict(event="meal_on_physical_tray", elapsed_sim_s=elapsed,
                                        tray_world_xyz=list(current_tray), release_drop_m=self.drop_m))
        self.ready = self._loaded and progress >= 1.
        row = dict(elapsed_sim_s=elapsed, phase=phase, ready=self.ready, tray_drift_m=drift,
                   tray_world_xyz=list(current_tray), wrist_receipt=receipt)
        self.records.append(row)
        return row

    def clear_performer(self):
        """Cut the presentation actor before collecting the clean navigation field."""
        if not self.ready or self._performer_cleared:
            raise RuntimeError("Performer cut requires one completed loading sequence")
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            prim = self.stage.GetPrimAtPath(self.server["root_path"])
            if not prim or not prim.IsActive():
                raise RuntimeError("The bar performer is missing before the presentation cut")
            prim.SetActive(False)
        self._performer_cleared = True
        event = dict(event="bar_performer_cut_before_clean_reference", elapsed_sim_s=self.elapsed,
                     performer_path=self.server["root_path"], meal_remains_on_physical_tray=True)
        self.events.append(event)
        return event
